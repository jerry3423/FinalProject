/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


/*
 *  Implement ray tracing using Ray-Query in a compute shader.
 *  This allows to compare the speed with RTX, but also to easier debug the shading pipeline,
 *  as it is not building a Mega kernel as the RTX pipeline does.
 */



#include "nvh/alignment.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "rayquery.hpp"
#include "scene.hpp"
#include "tools.hpp"

// Shaders
#include "autogen/pathtrace.comp.h"
#include "autogen/direct.comp.h"
#include "autogen/indirect.comp.h"
#include "autogen/denoiseDirect.comp.h"
#include "autogen/denoiseIndirect.comp.h"
#include "autogen/compose.comp.h"
//--------------------------------------------------------------------------------------------------
//
//
void RayQuery::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator)
{
  m_device     = device;
  m_pAlloc     = allocator;
  m_queueIndex = familyIndex;
  m_debug.setup(device);
}

//--------------------------------------------------------------------------------------------------
//
//
void RayQuery::destroy()
{
	for (int i = 0; i < 2; i++) {
		m_pAlloc->destroy(m_gbuffer[i]);
		m_pAlloc->destroy(m_directReservoir[i]);
		m_pAlloc->destroy(m_indirectReservoir[i]);
		m_pAlloc->destroy(m_denoiseTempBuf[i]);
		m_pAlloc->destroy(m_denoiseTempBuf[i + 2]);
	}
	m_pAlloc->destroy(m_directTempResv);
	m_pAlloc->destroy(m_indirectTempResv);
	m_pAlloc->destroy(m_motionVector);

	vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
	vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

	vkDestroyPipeline(m_device, m_directPipeline, nullptr);
	m_directPipeline = VK_NULL_HANDLE;
	vkDestroyPipeline(m_device, m_indirectPipeline, nullptr);
	m_indirectPipeline = VK_NULL_HANDLE;
	vkDestroyPipeline(m_device, m_denoiseDirectPipeline, nullptr);
	m_denoiseDirectPipeline = VK_NULL_HANDLE;
	vkDestroyPipeline(m_device, m_denoiseIndirectPipeline, nullptr);
	m_denoiseIndirectPipeline = VK_NULL_HANDLE;
	vkDestroyPipeline(m_device, m_composePipeline, nullptr);
	m_composePipeline = VK_NULL_HANDLE;
	vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);

}

//--------------------------------------------------------------------------------------------------
// Creation of the RQ pipeline
//
void RayQuery::create(const VkExtent2D& size, std::vector<VkDescriptorSetLayout> rtDescSetLayouts, Scene* scene)
{
	m_size = size;
	MilliTimer timer;
	LOGI("Create Ray Query Pipeline");

	std::vector<VkPushConstantRange> push_constants;
	push_constants.push_back({ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState) });

	// Create Gbuffer
	createImage();
	createBuffer();

	createDescriptorSet();
	rtDescSetLayouts.push_back(m_descSetLayout);

	VkPipelineLayoutCreateInfo layout_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	layout_info.pushConstantRangeCount = static_cast<uint32_t>(push_constants.size());
	layout_info.pPushConstantRanges = push_constants.data();
	layout_info.setLayoutCount = static_cast<uint32_t>(rtDescSetLayouts.size());
	layout_info.pSetLayouts = rtDescSetLayouts.data();
	vkCreatePipelineLayout(m_device, &layout_info, nullptr, &m_pipelineLayout);

	VkComputePipelineCreateInfo computePipelineCreateInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	computePipelineCreateInfo.layout = m_pipelineLayout;
	computePipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	computePipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	computePipelineCreateInfo.stage.pName = "main";

	computePipelineCreateInfo.stage.module = nvvk::createShaderModule(m_device, direct_comp, sizeof(direct_comp));
	vkCreateComputePipelines(m_device, {}, 1, &computePipelineCreateInfo, nullptr, &m_directPipeline);
	m_debug.setObjectName(m_directPipeline, "DirectPipeline");
	vkDestroyShaderModule(m_device, computePipelineCreateInfo.stage.module, nullptr);

	computePipelineCreateInfo.stage.module = nvvk::createShaderModule(m_device, indirect_comp, sizeof(indirect_comp));
	vkCreateComputePipelines(m_device, {}, 1, &computePipelineCreateInfo, nullptr, &m_indirectPipeline);
	m_debug.setObjectName(m_indirectPipeline, "IndirectPipeline");
	vkDestroyShaderModule(m_device, computePipelineCreateInfo.stage.module, nullptr);

	computePipelineCreateInfo.stage.module = nvvk::createShaderModule(m_device, denoiseDirect_comp, sizeof(denoiseDirect_comp));
	vkCreateComputePipelines(m_device, {}, 1, &computePipelineCreateInfo, nullptr, &m_denoiseDirectPipeline);
	m_debug.setObjectName(m_denoiseDirectPipeline, "denoiseDirectPipeline");
	vkDestroyShaderModule(m_device, computePipelineCreateInfo.stage.module, nullptr);

	computePipelineCreateInfo.stage.module = nvvk::createShaderModule(m_device, denoiseIndirect_comp, sizeof(denoiseIndirect_comp));
	vkCreateComputePipelines(m_device, {}, 1, &computePipelineCreateInfo, nullptr, &m_denoiseIndirectPipeline);
	m_debug.setObjectName(m_denoiseIndirectPipeline, "denoiseIndirectPipeline");
	vkDestroyShaderModule(m_device, computePipelineCreateInfo.stage.module, nullptr);

	computePipelineCreateInfo.stage.module = nvvk::createShaderModule(m_device, compose_comp, sizeof(compose_comp));
	vkCreateComputePipelines(m_device, {}, 1, &computePipelineCreateInfo, nullptr, &m_composePipeline);
	m_debug.setObjectName(m_composePipeline, "composePipeline");
	vkDestroyShaderModule(m_device, computePipelineCreateInfo.stage.module, nullptr);

	timer.print();
}


//--------------------------------------------------------------------------------------------------
// Executing the Ray Query compute shader
//
#define GROUP_SIZE 8  // Same group size as in compute shader

void RayQuery::run(const VkCommandBuffer& cmdBuf, const RtxState& state, nvvk::ProfilerVK& profiler, std::vector<VkDescriptorSet> descSets, int frames)
{

	RtxState cState = state;
	descSets.push_back(m_descSet[(frames + 1) % 2]);
	vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,static_cast<uint32_t>(descSets.size()), descSets.data(), 0, nullptr);

	vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState), &state);

	//Compute direct
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_directPipeline);
	vkCmdDispatch(cmdBuf, CEIL_DIV(state.size[0], BlockSizeX), CEIL_DIV(state.size[1], BlockSizeY), 1);

	//Compute indirect
	ivec2 indSize = state.size / 2;
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectPipeline);
	vkCmdDispatch(cmdBuf, CEIL_DIV(indSize[0], BlockSizeX), CEIL_DIV(indSize[1], BlockSizeY), 1);

	//Direct and indirect denoise stage
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_denoiseDirectPipeline);
	if (state.denoise > 0) {
		for (int i = 0; i < 4; i++) {
			cState.denoiseLevel = i;
			vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState), &cState);
			vkCmdDispatch(cmdBuf, CEIL_DIV(state.size[0], BlockSizeX), CEIL_DIV(state.size[1], BlockSizeY), 1);
		}

	}

	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_denoiseIndirectPipeline);
	if (state.denoise > 0) {
		for (int i = 0; i < 5; i++) {
			cState.denoiseLevel = i;
			vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState), &cState);
			vkCmdDispatch(cmdBuf, CEIL_DIV(indSize[0], BlockSizeX), CEIL_DIV(indSize[1], BlockSizeY), 1);
		}
	}

	//Compose stage
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_composePipeline);
	vkCmdDispatch(cmdBuf, CEIL_DIV(state.size[0], BlockSizeX), CEIL_DIV(state.size[1], BlockSizeY), 1);
}

void RayQuery::update(const VkExtent2D& size)
{
	m_size = size;
	for (int i = 0; i < 2; i++) {
		m_pAlloc->destroy(m_gbuffer[i]);
		m_pAlloc->destroy(m_directReservoir[i]);
		m_pAlloc->destroy(m_indirectReservoir[i]);
		m_pAlloc->destroy(m_denoiseTempBuf[i]);
		m_pAlloc->destroy(m_denoiseTempBuf[i + 2]);
	}
	m_pAlloc->destroy(m_directTempResv);
	m_pAlloc->destroy(m_indirectTempResv);
	m_pAlloc->destroy(m_motionVector);

	createImage();
	createBuffer();
	updateDescriptorSet();
}

void RayQuery::createBuffer()
{
	VkDeviceSize directSize = m_size.width * m_size.height * sizeof(DirectReservoir);
	VkDeviceSize indirectSize = (m_size.width / 2) * (m_size.height / 2) * sizeof(IndirectReservoir);
	for (int i = 0; i < 2; i++) {
		m_directReservoir[i] = m_pAlloc->createBuffer(directSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		m_indirectReservoir[i] = m_pAlloc->createBuffer(indirectSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	}
	m_directTempResv = m_pAlloc->createBuffer(directSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	m_indirectTempResv = m_pAlloc->createBuffer(indirectSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

void RayQuery::createImage()
{
	// Creating the color image
	{
		auto colorCreateInfo = nvvk::makeImage2DCreateInfo(
			m_size, m_gbufferFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true);

		nvvk::Image gbimage1 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(gbimage1.image);
		nvvk::Image gbimage2 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(gbimage2.image);
		VkImageViewCreateInfo ivInfo1 = nvvk::makeImageViewCreateInfo(gbimage1.image, colorCreateInfo);
		VkImageViewCreateInfo ivInfo2 = nvvk::makeImageViewCreateInfo(gbimage2.image, colorCreateInfo);

		m_gbuffer[0] = m_pAlloc->createTexture(gbimage1, ivInfo1);
		m_gbuffer[1] = m_pAlloc->createTexture(gbimage2, ivInfo2);
		m_gbuffer[0].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_gbuffer[1].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		auto motionVecCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_motionVectorFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
		nvvk::Image motionVecImg = m_pAlloc->createImage(motionVecCreateInfo);
		NAME_VK(motionVecImg.image);

		VkImageViewCreateInfo mvivInfo = nvvk::makeImageViewCreateInfo(motionVecImg.image, motionVecCreateInfo);
		m_motionVector = m_pAlloc->createTexture(motionVecImg, mvivInfo);
		m_motionVector.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		auto denoiseBufInfo = nvvk::makeImage2DCreateInfo(m_size, m_denoiseTempFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, false);
		nvvk::Image denoiseDirImageA = m_pAlloc->createImage(denoiseBufInfo);
		nvvk::Image denoiseDirImageB = m_pAlloc->createImage(denoiseBufInfo);
		nvvk::Image denoiseDirImageC = m_pAlloc->createImage(denoiseBufInfo);
		nvvk::Image denoiseDirImageD = m_pAlloc->createImage(denoiseBufInfo);
		VkImageViewCreateInfo divInfo1 = nvvk::makeImageViewCreateInfo(denoiseDirImageA.image, denoiseBufInfo);
		VkImageViewCreateInfo divInfo2 = nvvk::makeImageViewCreateInfo(denoiseDirImageB.image, denoiseBufInfo);
		VkImageViewCreateInfo divInfo3 = nvvk::makeImageViewCreateInfo(denoiseDirImageC.image, denoiseBufInfo);
		VkImageViewCreateInfo divInfo4 = nvvk::makeImageViewCreateInfo(denoiseDirImageD.image, denoiseBufInfo);

		m_denoiseTempBuf[0] = m_pAlloc->createTexture(denoiseDirImageA, divInfo1);
		m_denoiseTempBuf[1] = m_pAlloc->createTexture(denoiseDirImageB, divInfo2);
		m_denoiseTempBuf[2] = m_pAlloc->createTexture(denoiseDirImageC, divInfo3);
		m_denoiseTempBuf[3] = m_pAlloc->createTexture(denoiseDirImageD, divInfo4);
		m_denoiseTempBuf[0].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_denoiseTempBuf[1].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_denoiseTempBuf[2].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_denoiseTempBuf[3].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	}

	// Setting the image layout for both color and depth
	{
		nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
		auto              cmdBuf = genCmdBuf.createCommandBuffer();
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_motionVector.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_denoiseTempBuf[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_denoiseTempBuf[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_denoiseTempBuf[2].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_denoiseTempBuf[3].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

		genCmdBuf.submitAndWait(cmdBuf);
	}
}

void RayQuery::createDescriptorSet()
{
	m_bind = nvvk::DescriptorSetBindings{};

	VkShaderStageFlags flag = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
		| VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	m_bind.addBinding({ RayQBindings::eLastGbuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ RayQBindings::eThisGbuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });

	m_bind.addBinding({ RayQBindings::eLastDirectResv, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ RayQBindings::eThisDirectResv, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ RayQBindings::eTempDirectResv, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });

	m_bind.addBinding({ RayQBindings::eLastIndirectResv, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ RayQBindings::eThisIndirectResv, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ RayQBindings::eTempIndirectResv, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });

	m_bind.addBinding({ RayQBindings::eMotionVector, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });

	m_bind.addBinding({ RayQBindings::eDenoiseDirTempA, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ RayQBindings::eDenoiseDirTempB, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ RayQBindings::eDenoiseIndTempA, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ RayQBindings::eDenoiseIndTempB, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });

	m_descPool = m_bind.createPool(m_device, m_descSet.size());
	CREATE_NAMED_VK(m_descSetLayout, m_bind.createLayout(m_device));
	CREATE_NAMED_VK(m_descSet[0], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));
	CREATE_NAMED_VK(m_descSet[1], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));

	updateDescriptorSet();
}

void RayQuery::updateDescriptorSet()
{
	std::array<VkWriteDescriptorSet, 13> writes;
	VkDeviceSize directResvSize = m_size.width * m_size.height * sizeof(DirectReservoir);
	VkDeviceSize indirectResvSize = (m_size.width / 2) * (m_size.height / 2) * sizeof(IndirectReservoir);

	for (int i = 0; i < 2; i++) {
		VkDescriptorBufferInfo lastDirectResvBufInfo = { m_directReservoir[i].buffer, 0, directResvSize };
		VkDescriptorBufferInfo thisDirectResvBufInfo = { m_directReservoir[!i].buffer, 0, directResvSize };
		VkDescriptorBufferInfo tempDirectResvBufInfo = { m_directTempResv.buffer, 0, directResvSize };

		VkDescriptorBufferInfo lastIndirectResvBufInfo = { m_indirectReservoir[i].buffer, 0, indirectResvSize };
		VkDescriptorBufferInfo thisIndirectResvBufInfo = { m_indirectReservoir[!i].buffer, 0, indirectResvSize };
		VkDescriptorBufferInfo tempIndirectResvBufInfo = { m_indirectTempResv.buffer, 0, indirectResvSize };

		writes[0] = m_bind.makeWrite(m_descSet[i], RayQBindings::eLastGbuffer, &m_gbuffer[i].descriptor);
		writes[1] = m_bind.makeWrite(m_descSet[i], RayQBindings::eThisGbuffer, &m_gbuffer[!i].descriptor);

		writes[2] = m_bind.makeWrite(m_descSet[i], RayQBindings::eLastDirectResv, &lastDirectResvBufInfo);
		writes[3] = m_bind.makeWrite(m_descSet[i], RayQBindings::eThisDirectResv, &thisDirectResvBufInfo);
		writes[4] = m_bind.makeWrite(m_descSet[i], RayQBindings::eTempDirectResv, &tempDirectResvBufInfo);

		writes[5] = m_bind.makeWrite(m_descSet[i], RayQBindings::eLastIndirectResv, &lastIndirectResvBufInfo);
		writes[6] = m_bind.makeWrite(m_descSet[i], RayQBindings::eThisIndirectResv, &thisIndirectResvBufInfo);
		writes[7] = m_bind.makeWrite(m_descSet[i], RayQBindings::eTempIndirectResv, &tempIndirectResvBufInfo);

		writes[8] = m_bind.makeWrite(m_descSet[i], RayQBindings::eMotionVector, &m_motionVector.descriptor);

		writes[9] = m_bind.makeWrite(m_descSet[i], RayQBindings::eDenoiseDirTempA, &m_denoiseTempBuf[0].descriptor);
		writes[10] = m_bind.makeWrite(m_descSet[i], RayQBindings::eDenoiseDirTempB, &m_denoiseTempBuf[1].descriptor);
		writes[11] = m_bind.makeWrite(m_descSet[i], RayQBindings::eDenoiseIndTempA, &m_denoiseTempBuf[2].descriptor);
		writes[12] = m_bind.makeWrite(m_descSet[i], RayQBindings::eDenoiseIndTempB, &m_denoiseTempBuf[3].descriptor);

		vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	}
}


