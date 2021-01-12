#pragma once

#include <cstdint>
#include <vector>

#include "hash.h"

namespace stereo_vis {

	// Identifies a frame (scene), that consists of multiple images, 
	// a stereo pair in our case
	using FrameId = int64_t;

	// Identify the camera (left or right)
	using CamId = std::size_t;

	struct FrameCamId {
		FrameCamId() : frame_id(0), cam_id(0) {};

		FrameCamId(const FrameId& new_frame_id, const CamId& new_cam_id)
			: frame_id(new_frame_id), cam_id(new_cam_id) {};

		// Frame id or number of scenes
		FrameId frame_id;

		// camera id
		// 0 - left
		// 1 - right
		CamId cam_id;

		bool operator==(const FrameCamId& other) const {
			return (frame_id == other.frame_id) && (cam_id == other.cam_id);
		}

		bool operator!=(const FrameCamId& other) const {
			return (frame_id != other.frame_id) || (cam_id != other.cam_id);
		}

		bool operator <(const FrameCamId& other) const {
			if (frame_id == other.frame_id) return cam_id < other.cam_id;
			return frame_id < other.frame_id;
		}

		// for unorderd_map hashing
		explicit operator size_t() const {
			size_t seed = 0;
			hash_combine(seed, frame_id);
			hash_combine(seed, cam_id);
			return seed;
		}
	};
}

namespace std
{
	template<>
	struct hash < stereo_vis::FrameCamId > {
		inline std::size_t operator() (const stereo_vis::FrameCamId& val) const noexcept {
			std::size_t seed = 0;
			stereo_vis::hash_combine(seed, val.frame_id);
			stereo_vis::hash_combine(seed, val.cam_id);
			return seed;
		}
	};
}