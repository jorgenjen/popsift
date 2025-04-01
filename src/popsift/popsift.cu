/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/debug_macros.h"
#include "common/plane_2d.h"
#include "gauss_filter.h"
#include "popsift.h"
#include "s_image.h"
#include "sift_config.h"
#include "sift_pyramid.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace std;

PopSift::PopSift(const popsift::Config& config, popsift::Config::ProcessingMode mode, ImageMode imode, int device)
  : _image_mode(imode)
  , _device(device)
{
    cudaSetDevice(_device);
    configure(config);

    if(imode == ByteImages)
    {
        _pipe._unused.push(new popsift::Image);
        _pipe._unused.push(new popsift::Image);
    }
    else
    {
        _pipe._unused.push(new popsift::ImageFloat);
        _pipe._unused.push(new popsift::ImageFloat);
    }

    _pipe._thread_stage1.reset(new std::thread(&PopSift::uploadImages, this));
    if(mode == popsift::Config::ExtractingMode)
        _pipe._thread_stage2.reset(new std::thread(&PopSift::extractDownloadLoop, this));
    else
        _pipe._thread_stage2.reset(new std::thread(&PopSift::matchPrepareLoop, this));
}

PopSift::PopSift(ImageMode imode, int device)
  : _image_mode(imode)
  , _device(device)
{
    cudaSetDevice(_device);

    if(imode == ByteImages)
    {
        _pipe._unused.push(new popsift::Image);
        _pipe._unused.push(new popsift::Image);
    }
    else
    {
        _pipe._unused.push(new popsift::ImageFloat);
        _pipe._unused.push(new popsift::ImageFloat);
    }

    _pipe._thread_stage1.reset(new std::thread(&PopSift::uploadImages, this));
    _pipe._thread_stage2.reset(new std::thread(&PopSift::extractDownloadLoop, this));
}

PopSift::~PopSift()
{
    if(_isInit)
    {
        uninit();
    }
}

bool PopSift::configure(const popsift::Config& config, bool /*force*/)
{
    if(_pipe._pyramid != nullptr)
    {
        return false;
    }

    _config = config;
    _config.levels = max(2, config.levels);

    return true;
}

bool PopSift::applyConfiguration(bool force)
{
    if(force || (_config != _shadow_config))
    {
        cout << "\n\n\t\tApplying configuration nuuuuuu!!\n\n" << endl;

        popsift::init_filter(_config, _config.sigma, _config.levels);
        popsift::init_constants(_config.sigma,
                                _config.levels,
                                _config.getPeakThreshold(),
                                _config._edge_limit,
                                _config.getMaxExtrema(),
                                _config.getNormalizationMultiplier());
    }
    _shadow_config = _config;
    return true;
}

void PopSift::private_apply_scale_factor(int& w, int& h)
{
    /* up=-1 -> scale factor=2
     * up= 0 -> scale factor=1
     * up= 1 -> scale factor=0.5
     */
    float upscaleFactor = _config.getUpscaleFactor();
    float scaleFactor = 1.0f / powf(2.0f, -upscaleFactor);

    if(_config.octaves < 0)
    {
        int oct = max(int(floor(logf((float)min(w, h)) / logf(2.0f)) - 3.0f + scaleFactor), 1);
        _config.octaves = oct;
    }

    w = ceilf(w * scaleFactor);
    h = ceilf(h * scaleFactor);
}

bool PopSift::private_init(int w, int h)
{
    Pipe& p = _pipe;

    private_apply_scale_factor(w, h);

    if(p._pyramid != nullptr)
    {
        p._pyramid->resetDimensions(_config, w, h);
        return true;
    }

    p._pyramid = new popsift::Pyramid(_config, w, h);

    cudaDeviceSynchronize();

    return true;
}

bool PopSift::private_uninit()
{
    Pipe& p = _pipe;

    delete p._pyramid;
    p._pyramid = nullptr;

    return true;
}

void PopSift::uninit()
{
    if(!_isInit)
    {
        std::cerr << "[warning] Attempt to release resources from an uninitialized instance" << std::endl;
        return;
    }
    _pipe.uninit();

    _isInit = false;
}

PopSift::AllocTest PopSift::testTextureFit(int width, int height)
{
    const bool warn = popsift::cuda::device_prop_t::dont_warn;
    bool retval = _device_properties.checkLimit_2DtexLinear(width, height, warn);
    if(!retval)
    {
        return AllocTest::ImageExceedsLinearTextureLimit;
    }

    /* Scale the width and height - we need that size for the largest
     * octave. */
    private_apply_scale_factor(width, height);

    /* _config.level does not contain the 3 blur levels beyond the first
     * that is required for downscaling to the following octave.
     * We need all layers to check if we can support enough layers.
     */
    int depth = _config.levels + 3;

    retval = _device_properties.checkLimit_2DsurfLayered(width, height, depth, warn);

    return (retval ? AllocTest::Ok : AllocTest::ImageExceedsLayeredSurfaceLimit);
}

std::string PopSift::testTextureFitErrorString(AllocTest err, int width, int height)
{
    ostringstream ostr;

    switch(err)
    {
        case AllocTest::Ok: ostr << "?    No error." << endl; break;
        case AllocTest::ImageExceedsLinearTextureLimit:
            _device_properties.checkLimit_2DtexLinear(width, height, false);
            ostr << "E    Cannot load unscaled image. " << endl
                 << "E    It exceeds the max CUDA linear texture size. " << endl
                 << "E    Max is (" << width << "," << height << ")" << endl;
            break;
        case AllocTest::ImageExceedsLayeredSurfaceLimit: {
            const float upscaleFactor = _config.getUpscaleFactor();
            const float scaleFactor = 1.0f / powf(2.0f, -upscaleFactor);
            int w = ceilf(width * scaleFactor);
            int h = ceilf(height * scaleFactor);
            int d = _config.levels + 3;

            _device_properties.checkLimit_2DsurfLayered(w, h, d, false);

            w = w / scaleFactor;
            h = h / scaleFactor;
            ostr << "E    Cannot use" << (upscaleFactor == 1 ? " default " : " ") << "downscaling factor "
                 << -upscaleFactor << " (i.e. upscaling by " << pow(2, upscaleFactor) << "). " << endl
                 << "E    It exceeds the max CUDA layered surface size. " << endl
                 << "E    Change downscaling to fit into (" << w << "," << h << ") with " << (d - 3)
                 << " levels per octave." << endl;
        }
        break;
        default: ostr << "E    Programming error, please report." << endl; break;
    }
    return ostr.str();
}

SiftJob* PopSift::enqueue(int w, int h, const unsigned char* imageData)
{
    if(_image_mode != ByteImages)
    {
        stringstream ss;
        ss << "Image mode error" << endl
           << "E    Cannot load byte images into a PopSift pipeline configured for float images";
        POP_FATAL(ss.str());
    }

    AllocTest a = testTextureFit(w, h); // does not modify my w and h but uses sclaed up for texture fit check
    if(a != AllocTest::Ok)
    {
        cerr << __FILE__ << ":" << __LINE__ << " Image too large" << endl << testTextureFitErrorString(a, w, h);
        return nullptr;
    }

    SiftJob* job = new SiftJob(w, h, imageData); // still normal w and h at this point
    _pipe._queue_stage1.push(job);
    return job;
}

SiftJob* PopSift::enqueue(int w, int h, const float* imageData)
{
    if(_image_mode != FloatImages)
    {
        stringstream ss;
        ss << "Image mode error" << endl
           << "E    Cannot load float images into a PopSift pipeline configured for byte images";
        POP_FATAL(ss.str());
    }

    AllocTest a = testTextureFit(w, h);
    if(a != AllocTest::Ok)
    {
        cerr << __FILE__ << ":" << __LINE__ << " Image too large" << endl << testTextureFitErrorString(a, w, h);
        return nullptr;
    }

    SiftJob* job = new SiftJob(w, h, imageData);
    _pipe._queue_stage1.push(job);
    return job;
}

void PopSift::uploadImages()
{
    cudaSetDevice(_device);

    SiftJob* job;
    while((job = _pipe._queue_stage1.pull()) != nullptr)
    {
        popsift::ImageBase* img = _pipe._unused.pull();
        job->setImg(img);
        _pipe._queue_stage2.push(job);
    }
    _pipe._queue_stage2.push(nullptr);
}

void PopSift::extractDownloadLoop()
{
    cudaSetDevice(_device);
    applyConfiguration(true);

    Pipe& p = _pipe;

    SiftJob* job;
    while((job = p._queue_stage2.pull()) != nullptr)
    {
        applyConfiguration();

        popsift::ImageBase* img = job->getImg();

        private_init(img->getWidth(), img->getHeight());

        p._pyramid->step1(_config, img);
        p._unused.push(img); // uploaded input image no longer needed, release for reuse

        p._pyramid->step2(_config);

        popsift::FeaturesHost* features = p._pyramid->get_descriptors(_config);

        cudaDeviceSynchronize();

        bool log_to_file = (_config.getLogMode() == popsift::Config::All);
        if(log_to_file)
        {
            // int octaves = p._pyramid->getNumOctaves();
            // for( int o=0; o<octaves; o++ ) { p._pyramid->download_descriptors( _config, o ); }
            // int levels  = p._pyramid->getNumLevels();

            p._pyramid->download_and_save_array("pyramid");
            p._pyramid->save_descriptors(_config, features, "pyramid");
        }

        job->setFeatures(features);
    }

    private_uninit();
}

void PopSift::matchPrepareLoop()
{
    cudaSetDevice(_device);
    applyConfiguration(true);

    Pipe& p = _pipe;

    SiftJob* job;
    while((job = p._queue_stage2.pull()) != nullptr)
    {
        popsift::FeaturesDev* features;
        try
        {
            applyConfiguration();

            popsift::ImageBase* img = job->getImg();

            private_init(img->getWidth(), img->getHeight());

            p._pyramid->step1(_config, img);
            p._unused.push(img); // uploaded input image no longer needed, release for reuse

            p._pyramid->step2(_config);

            features = p._pyramid->clone_device_descriptors(_config);
            cudaDeviceSynchronize();
        }
        catch(const std::exception& e)
        {
            job->setError(std::current_exception());
            job->setFeatures(nullptr);
            break;
        }

        job->setFeatures(features);
    }

    private_uninit();
}

SiftJob::SiftJob(int w, int h, const unsigned char* imageData)
  : _w(w)
  , _h(h)
  , _img(nullptr)
{
    _f = _p.get_future();

    _imageData = (unsigned char*)malloc(w * h);
    if(_imageData != nullptr)
    {
        memcpy(_imageData, imageData, w * h);
    }
    else
    {
        stringstream ss;
        ss << "Memory limitation" << endl << "E    Failed to allocate memory for SiftJob";
        POP_FATAL(ss.str());
    }
}

SiftJob::SiftJob(int w, int h, const float* imageData)
  : _w(w)
  , _h(h)
  , _img(nullptr)
{
    _f = _p.get_future();

    _imageData = (unsigned char*)malloc(w * h * sizeof(float));
    if(_imageData != nullptr)
    {
        memcpy(_imageData, imageData, w * h * sizeof(float));
    }
    else
    {
        stringstream ss;
        ss << "Memory limitation" << endl << "E    Failed to allocate memory for SiftJob";
        POP_FATAL(ss.str());
    }
}

SiftJob::~SiftJob() { free(_imageData); }

__global__ static void printTexturePixels(cudaTextureObject_t texObj)
{
    for(int y = 0; y < 10; ++y)
    {
        for(int x = 0; x < 10; ++x)
        {
            // Normalize coordinates to [0, 1) range
            float u = (x + 0.5f) / 1280.f;
            float v = (y + 0.5f) / 852.f;

            // float u = (x) / 640.0f;
            // float v = (y) / 640.0f;
            // Sample the texture
            float pixel = tex2D<float>(texObj, u, v);

            // Print the pixel value
            printf("%06.2f ", pixel * 255.0f);
        }
        printf("\n");
    }
}

void SiftJob::setImg(popsift::ImageBase* img)
{
    img->resetDimensions(_w, _h);
    img->load(_imageData);

    // test _imageDate and then the host plane (texture)
    printf("Width: %d", _w);
    for(int y = 0; y < 5; ++y)
    {
        for(int x = 0; x < 5; ++x)
        {
            // printf("\t\tValue at %d %d: %f\n", x, y, tex2D<float>(src_linear_tex, x, y));
            // printf("\t\tPixel val(%d, %d): %d\n", x, y, _imageData[x + y * _w]);
        }
    }
    popsift::Image* me_img = (popsift::Image*)img;

    popsift::Plane2D_uint8 h_p = me_img->_input_image_h;

    uint8_t* plane_img = h_p.data;

    // plane on host
    for(int y = 0; y < 5; ++y)
    {
        for(int x = 0; x < 5; ++x)
        {
            // printf("\t\tValue at %d %d: %f\n", x, y, tex2D<float>(src_linear_tex, x, y));
            // printf("\t\tPixel val ptr (%d, %d): %d\n", x, y, _imageData[x + y * _w]);
            // still not pitch aligned which is 1024(for 640 _w) still based on width of the iamge
            // printf("\t\tPixel val tex (%d, %d): %d\n\n", x, y, plane_img[x + y * _w]);
        }
    }

    // print out the

    // printTexturePixels<<<1, 1>>>(img->getInputTexture());

    _img = img;
}

popsift::ImageBase* SiftJob::getImg() { return _img; }

void SiftJob::setFeatures(popsift::FeaturesBase* f) { _p.set_value(f); }

popsift::FeaturesHost* SiftJob::get() { return getHost(); }

popsift::FeaturesBase* SiftJob::getBase() { return _f.get(); }

popsift::FeaturesHost* SiftJob::getHost() { return dynamic_cast<popsift::FeaturesHost*>(_f.get()); }

popsift::FeaturesDev* SiftJob::getDev()
{
    popsift::FeaturesBase* features = _f.get();
    if(this->_err != nullptr)
    {
        std::rethrow_exception(this->_err);
    }
    return dynamic_cast<popsift::FeaturesDev*>(features);
}

void SiftJob::setError(std::exception_ptr ptr) { this->_err = ptr; }

void PopSift::Pipe::uninit()
{
    _queue_stage1.push(nullptr);
    if(_thread_stage2 != nullptr)
    {
        _thread_stage2->join();
        _thread_stage2.reset(nullptr);
    }
    if(_thread_stage1 != nullptr)
    {
        _thread_stage1->join();
        _thread_stage1.reset(nullptr);
    }

    while(!_unused.empty())
    {
        popsift::ImageBase* img = _unused.pull();
        delete img;
    }
}
