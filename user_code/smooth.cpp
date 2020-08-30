
#include <gflags/gflags.h>
// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif
// OpenPose dependencies
#include <openpose/core/array.hpp>
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/hand/headers.hpp>
#include <openpose/face/headers.hpp>
#include "utility.h"

DEFINE_int32(logging_level, 3, "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
	" 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
	" low priority messages and 4 for important ones.");

DEFINE_string(points_dir, "D:/download/jile/source_jilejingtu_pose_jsons_smoothed", "");
DEFINE_string(out_dir, "D:/download/jile/source_jilejingtu_pose_sticks_smoothed", "");

DEFINE_string(output_resolution, "1024x1920", "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
	" input image resolution.");
DEFINE_string(model_pose, "BODY_25", "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
	"`MPI_4_layers` (15 keypoints, even faster but less accurate).");
// OpenPose Rendering
DEFINE_bool(disable_blending, false, "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
	" background, instead of being rendered into the original image. Related: `part_to_show`,"
	" `alpha_pose`, and `alpha_pose`.");
DEFINE_double(render_threshold, 0.05, "Only estimated keypoints whose score confidences are higher than this threshold will be"
	" rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
	" while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
	" more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
	" hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_face, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
	" hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_hand, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
	" hide it. Only valid for GPU rendering.");

DEFINE_double(face_render_threshold, 0.4, "Analogous to `render_threshold`, but applied to the face keypoints.");
DEFINE_double(hand_render_threshold, 0.2, "Analogous to `render_threshold`, but applied to the hand keypoints.");
DEFINE_int32(render_mode, 1, "Set to 0 for no rendering, 1 for CPU rendering (slightly faster), and 2 for GPU rendering");

//DEFINE_double(pose_render_r_base, 100, "circleRadius_base.");
//DEFINE_double(pose_render_l_base, 120, "lineWidthBase.");
//DEFINE_double(face_render_r_base, 120, ".");
//DEFINE_double(face_render_l_base, 250, ".");
//DEFINE_double(hand_render_r_base, 100, ".");
//DEFINE_double(hand_render_l_base, 80, ".");

DEFINE_string(model_folder, "models/", "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_int32(num_gpu_start, 0, "GPU device start number.");
DEFINE_double(alpha_heatmap, 0.7, "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
	" heatmap, 0 will only show the frame. Only valid for GPU rendering.");

DEFINE_bool(face, false, ".");

void renderCPU() {
	cout << "render on CPU" << endl;
	vector<string> points_paths = getFiles(FLAGS_points_dir, true);
	op::OpOutputToCvMat opOutputToCvMat;
	if (access(FLAGS_out_dir.c_str(), 0) == -1)
	{
#ifdef linux
		int flag = mkdir(FLAGS_out_dir.c_str(), 0777);
		if (flag == 0) cout << "make successfully" << endl;
		else cout << "make errorly" << endl;
		CV_Assert(flag == 0);
#endif
#ifdef WIN
		int flag = mkdir(FLAGS_out_dir.c_str());
		if (flag == 0) cout << "make successfully" << endl;
		else cout << "make errorly" << endl;
		CV_Assert(flag == 0);
#endif
	}

	const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
	op::PoseCpuRenderer poseRenderer{ poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
		(float)FLAGS_alpha_pose };

	op::FaceCpuRenderer faceRenderer{ (float)FLAGS_face_render_threshold,(float)FLAGS_alpha_pose };

	op::HandCpuRenderer handRenderer{ (float)FLAGS_hand_render_threshold };

	const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");

	for (int i = 0; i < points_paths.size(); ++i) {
		string path = points_paths[i];
		vector<float> pose_points;
		vector<float> face_points;
		vector<float> hand_left_points;
		vector<float> hand_right_points;
		getPointsFromTxt(path, pose_points, face_points, hand_left_points, hand_right_points);
		vector<int> pose_size = { 1,25,3 };
		op::Array<float> poseKeypoints(pose_size, pose_points.data());

		vector<int> face_size = { 1,70,3 };
		op::Array<float> faceKeypoints(face_size, face_points.data());

		vector<int> hand_size = { 1,21,3 };
		op::Array<float> handLeftKeypoints(hand_size, hand_right_points.data());
		op::Array<float> handRightKeypoints(hand_size, hand_right_points.data());
		std::array<op::Array<float>, 2> handPoints = { handLeftKeypoints,handRightKeypoints };

		op::Array<float> outputArray({ outputSize.x,outputSize.y, 3 });
		poseRenderer.renderPose(outputArray, poseKeypoints, 1.0);

		faceRenderer.renderFace(outputArray, faceKeypoints, 1.0);

		handRenderer.renderHand(outputArray, handPoints, 1.0);
		auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);
		imshow("points", outputImage);
		waitKey(2);

		string image_path = FLAGS_out_dir + "/" + int2string(5, i) + ".png";
		imwrite(image_path, outputImage);
	}
}

void renderGPU() {
	cout << "render on GPU" << endl;
	vector<string> points_paths = getFiles(FLAGS_points_dir, true);
	op::OpOutputToCvMat opOutputToCvMat;

	if (access(FLAGS_out_dir.c_str(), 0) == -1)
	{
#ifdef linux
		int flag = mkdir(FLAGS_out_dir.c_str(), 0777);
		if (flag == 0) cout << "make successfully" << endl;
		else cout << "make errorly" << endl;
		CV_Assert(flag == 0);
#endif
#ifdef WIN
		int flag = mkdir(FLAGS_out_dir.c_str());
		if (flag == 0) cout << "make successfully" << endl;
		else cout << "make errorly" << endl;
		CV_Assert(flag == 0);
#endif
	}

	const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
	auto poseExtractorPtr = std::make_shared<op::PoseExtractorCaffe>(poseModel, FLAGS_model_folder,
		FLAGS_num_gpu_start);
	op::PoseGpuRenderer poseGpuRenderer{ poseModel, poseExtractorPtr, (float)FLAGS_render_threshold,
		!FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap};

	op::FaceGpuRenderer faceRenderer{ (float)FLAGS_face_render_threshold, (float)FLAGS_alpha_face,0.};

	op::HandGpuRenderer handRenderer{ (float)FLAGS_hand_render_threshold, (float)FLAGS_alpha_hand,0};

	poseExtractorPtr->initializationOnThread();
	poseGpuRenderer.initializationOnThread();
	faceRenderer.initializationOnThread();
	handRenderer.initializationOnThread();

	const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");

	for (int i = 0; i < points_paths.size(); ++i) {
		string path = points_paths[i];
		cout << path << endl;
		vector<float> pose_points;
		vector<float> face_points;
		vector<float> hand_left_points;
		vector<float> hand_right_points;
		getPointsFromTxt(path, pose_points, face_points, hand_left_points, hand_right_points);
		vector<int> pose_size = { 1,25,3 };
		op::Array<float> poseKeypoints(pose_size, pose_points.data());

		vector<int> face_size = { 1,70,3 };
		op::Array<float> faceKeypoints(face_size, face_points.data());

		vector<int> hand_size = { 1,21,3 };
		op::Array<float> handLeftKeypoints(hand_size, hand_left_points.data());
		op::Array<float> handRightKeypoints(hand_size, hand_right_points.data());
		std::array<op::Array<float>, 2> handPoints = { handLeftKeypoints,handRightKeypoints };

		vector<float> image(outputSize.x*outputSize.y*3, 0);
		op::Array<float> outputArray({ outputSize.x,outputSize.y, 3 }, image.data());
		poseGpuRenderer.renderPose(outputArray, poseKeypoints, 1.0);
		if(FLAGS_face == true){
            faceRenderer.renderFace(outputArray, faceKeypoints, 1.0);
        }
		handRenderer.renderHand(outputArray, handPoints, 1.0);
		auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);
		string image_path = FLAGS_out_dir + "/" + int2string(5, i) + ".png";
		imwrite(image_path, outputImage);
	}
}

int main(int argc, char *argv[])
{
	// Parsing command line flags
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (FLAGS_render_mode == 1) {
		renderCPU();
	}
	else if (FLAGS_render_mode == 2) {
		renderGPU();
	}
	else {
		cout << "choose CPU or GPU" << endl;
	}

	return 0;
}
