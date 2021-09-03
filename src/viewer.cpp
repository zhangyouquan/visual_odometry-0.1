#include "viewer.h"
#include "feature.h"
#include "frame.h"

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

namespace myslam
{
    Viewer::Viewer()
    {
        //std::bind用来将可调用对象与其参数一起进行绑定。绑定结果可以用std::funtion保存，并延迟调用到任何我们需要的时候。
        viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop,this));
    }

    void Viewer::Close()
    {
        viewer_running_ = false;
        viewer_thread_.join();
    }

    void Viewer::AddCurrentFrame(Frame::Ptr currentframe)
    {
        std::unique_lock<std::mutex> lck(viewer_data_mutex_);
        current_frame_ = currentframe;
    }

    void Viewer::UpdataMap()
    {
        std::unique_lock<std::mutex> lck(viewer_data_mutex_);
        assert(map_ != nullptr);//检查一下地图是不是不为空
        active_keyframes_ = map_->GetActiveKeyFrames();
        active_landmarks_ = map_->GetActiveMapPoints();
        map_updated_ = true;
    }

    void Viewer::ThreadLoop()
    {
        pangolin::CreateWindowAndBind("Stere VO Frontend",1024,768);
        //开启更新缓冲区的功能，也就是说，如果深度值发生变化，会随时更新
        //glDisable(GL_DEPTH_TEST)是关闭此功能
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_CONSTANT_ALPHA);

        //定义目标并初始化
        pangolin::OpenGlRenderState vis_camera(
                //图像宽高，4个内参 和最近最远视距
                pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
                //相机所在坐标，视点所在坐标，图像的下方为y
                pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0)
                );

        //创建一个交互界面
        pangolin::Handler3D *handler = new pangolin::Handler3D(vis_camera);//创建一个相机
        pangolin::View& vis_display =
                pangolin::CreateDisplay()
                                    // 视图在视窗中的范围
                        .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                        .SetHandler(handler);

        const float blue[3] = {0, 0, 1};
        const float green[3] = {0, 1, 0};

        while (!pangolin::ShouldQuit() && viewer_running_)
        {
            //清空画面
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);//白底
            vis_display.Activate(vis_camera);//激活一个视角。

            std::unique_lock<std::mutex> lock(viewer_data_mutex_);
            if (current_frame_)
            {
                DrawFrame(current_frame_, green);
                FollowCurrentFrame(vis_camera);

                cv::Mat img = PlotFrameImage();
                cv::imshow("image", img);
                cv::waitKey(1);
            }

            // 画地图点
            if (map_)
            {
                DrawMapPoints();
            }

            pangolin::FinishFrame();
            usleep(5000);
        }
        std::cout << "Stop viewer";
    }

    void Viewer::DrawFrame(Frame::Ptr frame, const float *color)
    {
        SE3 Twc = frame->Pose().inverse();//从世界系看相机系，也就是说相机是动的
        const float sz = 1.0;
        const int line_width = 2.0;
        const float fx = 400;
        const float fy = 400;
        const float cx = 512;
        const float cy = 384;
        const float width = 1080;
        const float height = 768;

        glPushMatrix();//保存当前矩阵状态

        Sophus::Matrix4f m = Twc.matrix().template cast<float>();
        glMultMatrixf((GLfloat*)m.data());//进行一次欧式变换

        if (color == nullptr)
        {
            glColor3f(1, 0, 0);
        } else
            glColor3f(color[0], color[1], color[2]);

        glLineWidth(line_width);//设置线宽
        glBegin(GL_LINES);//2点确定一条直线，奇数点无效
        glVertex3f(0, 0, 0);//界面中心点
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
        //到此正好画好一个图像帧

        glEnd();
        glPopMatrix();//回到glPushMatrix()时状态
    }

    void Viewer::FollowCurrentFrame(pangolin::OpenGlRenderState &vis_camera)
    {
        SE3 Twc = current_frame_->Pose().inverse();
        pangolin::OpenGlMatrix m(Twc.matrix());
        vis_camera.Follow(m, true);
    }

    cv::Mat Viewer::PlotFrameImage()
    {
        cv::Mat img_out;
        cv::cvtColor(current_frame_->left_img_,img_out,CV_GRAY2BGR);
        for(size_t i = 0; i < current_frame_->features_left_.size(); i++)
        {
            if(current_frame_->features_left_.at(i)->map_point_.lock())
            {
                auto feat = current_frame_->features_left_.at(i);
                cv::circle(img_out,feat->position_.pt,2,cv::Scalar(0,255,0),2);
            }
        }
        return img_out;
    }

    void Viewer::DrawMapPoints()
    {
        const float red[3] ={1.0, 0, 0};
        for (auto& kf : active_keyframes_)
        {
            DrawFrame(kf.second, red);//历史关键帧都画为红色
        }

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto& landmark : active_landmarks_)
        {
            auto pos = landmark.second->GetPos();
            glColor3f(red[0], red[1], red[2]);
            glVertex3d(pos[0], pos[1], pos[2]);
        }
        glEnd();
    }
}