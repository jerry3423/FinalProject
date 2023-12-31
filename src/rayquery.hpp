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


#pragma once

#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"

#include "nvvk/profiler_vk.hpp"
#include "renderer.h"
#include "shaders/host_device.h"

/*

Creating the Compute ray query renderer 
* Requiring:  
  - Acceleration structure (AccelSctruct / Tlas)
  - An image (Post StoreImage)
  - The glTF scene (vertex, index, materials, ... )

* Usage
  - setup as usual
  - create
  - run
*/
class RayQuery : public Renderer
{
public:
  void setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator) override;
  void destroy() override;
  void create(const VkExtent2D& size, std::vector<VkDescriptorSetLayout> rtDescSetLayouts, Scene* scene) override;
  const std::string name() override { return std::string("RQ"); }
  void run(const VkCommandBuffer& cmdBuf, const RtxState& state, nvvk::ProfilerVK& profiler, std::vector<VkDescriptorSet> descSets, int frames) override;
  void update(const VkExtent2D& size) override;
  void createBuffer();
  void createImage();
  void createDescriptorSet();
  void updateDescriptorSet();

private:
  uint32_t m_nbHit{0};

private:
  // Setup
  nvvk::ResourceAllocator* m_pAlloc{nullptr};  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil          m_debug;            // Utility to name objects
  VkDevice                 m_device{VK_NULL_HANDLE};
  uint32_t                 m_queueIndex{0};

  std::array<nvvk::Texture, 2> m_gbuffer;

  std::array<nvvk::Buffer, 2> m_directReservoir;
  std::array<nvvk::Buffer, 2> m_indirectReservoir;

  nvvk::Buffer m_directTempResv;
  nvvk::Buffer m_indirectTempResv;

  // Depth 32bit, Normal 32bit, Metallic 8bit, Roughness 8bit, IOR 8bit, Transmission 8bit, Albedo 24bit, Hashed Material ID 8bit
  VkFormat m_gbufferFormat{ VK_FORMAT_R32G32B32A32_UINT };
  VkFormat m_denoiseTempFormat{ VK_FORMAT_R32G32B32A32_SFLOAT };

  nvvk::Texture m_motionVector;
  VkFormat m_motionVectorFormat{ VK_FORMAT_R16G16_SINT };


  VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};

  VkPipeline m_directPipeline{ VK_NULL_HANDLE };
  VkPipeline m_indirectPipeline{ VK_NULL_HANDLE };

  nvvk::DescriptorSetBindings m_bind;
  VkDescriptorPool      m_descPool{ VK_NULL_HANDLE };
  VkDescriptorSetLayout m_descSetLayout{ VK_NULL_HANDLE };
  std::array<VkDescriptorSet, 2> m_descSet{ VK_NULL_HANDLE };

  VkExtent2D m_size{};
  int m_frameInd = 0;
};
