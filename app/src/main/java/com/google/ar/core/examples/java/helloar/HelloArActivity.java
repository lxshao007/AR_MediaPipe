/*
 * Copyright 2017 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ar.core.examples.java.helloar;

import android.content.DialogInterface;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.SurfaceTexture;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.google.ar.core.Anchor;
import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.Coordinates2d;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Point;
import com.google.ar.core.Point.OrientationMode;
import com.google.ar.core.PointCloud;
import com.google.ar.core.Session;
import com.google.ar.core.Trackable;
import com.google.ar.core.TrackingState;
import com.google.ar.core.examples.java.common.helpers.CameraPermissionHelper;
import com.google.ar.core.examples.java.common.helpers.DepthSettings;
import com.google.ar.core.examples.java.common.helpers.DisplayRotationHelper;
import com.google.ar.core.examples.java.common.helpers.FullScreenHelper;
import com.google.ar.core.examples.java.common.helpers.SnackbarHelper;
import com.google.ar.core.examples.java.common.helpers.TapHelper;
import com.google.ar.core.examples.java.common.helpers.TrackingStateHelper;
import com.google.ar.core.examples.java.common.rendering.BackgroundRenderer;
import com.google.ar.core.examples.java.common.rendering.ObjectRenderer;
import com.google.ar.core.examples.java.common.rendering.ObjectRenderer.BlendMode;
import com.google.ar.core.examples.java.common.rendering.PlaneRenderer;
import com.google.ar.core.examples.java.common.rendering.PointCloudRenderer;
import com.google.ar.core.examples.java.common.rendering.Texture;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;
import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.glutil.EglManager;
import com.google.protobuf.InvalidProtocolBufferException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * This is a simple example that shows how to create an augmented reality (AR) application using the
 * ARCore API. The application will display any detected planes and will allow the user to tap on a
 * plane to place a 3d model of the Android robot.
 */
public class HelloArActivity extends AppCompatActivity {
  private static final String TAG = "HelloArActivity";
  private static final String BINARY_GRAPH_NAME ="hand_tracking_mobile_gpu.binarypb";
  private static final String OUTPUT_HAND_PRESENCE_STREAM_NAME = "hand_presence";
  private static final String OUTPUT_LANDMARKS_STREAM_NAME = "hand_landmarks";
  private static final String INPUT_VIDEO_STREAM_NAME = "input_video";
  private static final String OUTPUT_VIDEO_STREAM_NAME = "output_video";

  // Flips the camera-preview frames vertically by default, before sending them into FrameProcessor
  // to be processed in a MediaPipe graph, and flips the processed frames back when they are
  // displayed. This maybe needed because OpenGL represents images assuming the image origin is at
  // the bottom-left corner, whereas MediaPipe in general assumes the image origin is at the
  // top-left corner.
  // NOTE: use "flipFramesVertically" in manifest metadata to override this behavior.
  private static final boolean FLIP_FRAMES_VERTICALLY = true;

    static {
        // Load all native libraries needed by the app.
        System.loadLibrary("mediapipe_jni");
        try {
            System.loadLibrary("opencv_java3");
        } catch (java.lang.UnsatisfiedLinkError e) {
            // Some example apps (e.g. template matching) require OpenCV 4.
            System.loadLibrary("opencv_java4");
        }
    }

  // {@link SurfaceTexture} where the camera-preview frames can be accessed.
  private SurfaceTexture previewFrameTexture;
  // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
  private SurfaceView previewDisplayView;
  // Creates and manages an {@link EGLContext}.
  private EglManager eglManager;
  // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
  // frames onto a {@link Surface}.
  private FrameProcessor processor;
  // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
  // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
  private ExternalTextureConverter converter;
  // Handles camera access via the {@link CameraX} Jetpack support library.
  private CameraXPreviewHelper cameraHelper;


  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    previewDisplayView = new SurfaceView(this);
    setupPreviewDisplayView();

    // Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
    // binary graphs.
    AndroidAssetUtil.initializeNativeAssetManager(this);
    eglManager = new EglManager(null);
    processor =
            new FrameProcessor(
                    this,
                    eglManager.getNativeContext(),
                    BINARY_GRAPH_NAME,
                    INPUT_VIDEO_STREAM_NAME,
                    OUTPUT_VIDEO_STREAM_NAME);
    processor.getVideoSurfaceOutput().setFlipY(FLIP_FRAMES_VERTICALLY);
    processor.addPacketCallback(
            OUTPUT_LANDMARKS_STREAM_NAME,
            (packet) -> {
                Boolean handPresence = PacketGetter.getBool(packet);
                if (!handPresence) {
                    Log.d(
                            TAG,
                            "[TS:" + packet.getTimestamp() + "] Hand presence is false, no hands detected.");
                }
            });
      if (true) {
          processor.addPacketCallback(
                  OUTPUT_LANDMARKS_STREAM_NAME,
                  (packet) -> {
                      byte[] landmarksRaw = PacketGetter.getProtoBytes(packet);
                      try {
                          NormalizedLandmarkList landmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
                          if (landmarks == null) {
                              Log.v(TAG, "[TS:" + packet.getTimestamp() + "] No hand landmarks.");
                              return;
                          }
                          // Note: If hand_presence is false, these landmarks are useless.
                          Log.v(
                                  TAG,
                                  "[TS:"
                                          + packet.getTimestamp()
                                          + "] #Landmarks for hand: "
                                          + landmarks.getLandmarkCount());
                          Log.v(TAG, getLandmarksDebugString(landmarks));
                      } catch (InvalidProtocolBufferException e) {
                          Log.e(TAG, "Couldn't Exception received - " + e);
                          return;
                      }
                  });
      }
    PermissionHelper.checkAndRequestCameraPermissions(this);
  }

  private static String getLandmarksDebugString(NormalizedLandmarkList landmarks) {
    int landmarkIndex = 0;
    String landmarksString = "";
    for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
      landmarksString +=
              "\t\tLandmark["
                      + landmarkIndex
                      + "]: ("
                      + landmark.getX()
                      + ", "
                      + landmark.getY()
                      + ", "
                      + landmark.getZ()
                      + ")\n";
      ++landmarkIndex;
    }
    return landmarksString;
  }

  @Override
  protected void onResume() {
    super.onResume();
    converter = new ExternalTextureConverter(eglManager.getContext());
    converter.setFlipY(FLIP_FRAMES_VERTICALLY);
    converter.setConsumer(processor);
    if (PermissionHelper.cameraPermissionsGranted(this)) {
      startCamera();
    }
  }

  @Override
  protected void onPause() {
    super.onPause();
    converter.close();
  }

  @Override
  public void onRequestPermissionsResult(
          int requestCode, String[] permissions, int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);
  }
  private void setupPreviewDisplayView() {
    previewDisplayView.setVisibility(View.GONE);
    ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
    viewGroup.addView(previewDisplayView);
    previewDisplayView
            .getHolder()
            .addCallback(
                    new SurfaceHolder.Callback() {
                      @Override
                      public void surfaceCreated(SurfaceHolder holder) {
                        processor.getVideoSurfaceOutput().setSurface(holder.getSurface());
                      }
                      @Override
                      public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                        // (Re-)Compute the ideal size of the camera-preview display (the area that the
                        // camera-preview frames get rendered onto, potentially with scaling and rotation)
                        // based on the size of the SurfaceView that contains the display.
                        Size viewSize = new Size(width, height);
                        Size displaySize = cameraHelper.computeDisplaySizeFromViewSize(viewSize);
                        // Connect the converter to the camera-preview frames as its input (via
                        // previewFrameTexture), and configure the output width and height as the computed
                        // display size.
                        converter.setSurfaceTextureAndAttachToGLContext(
                                previewFrameTexture, displaySize.getWidth(), displaySize.getHeight());
                      }
                      @Override
                      public void surfaceDestroyed(SurfaceHolder holder) {
                        processor.getVideoSurfaceOutput().setSurface(null);
                      }
                    });
  }
  private void startCamera() {
    cameraHelper = new CameraXPreviewHelper();
    cameraHelper.setOnCameraStartedListener(
            surfaceTexture -> {
              previewFrameTexture = surfaceTexture;
              // Make the display view visible to start showing the preview. This triggers the
              // SurfaceHolder.Callback added to (the holder of) previewDisplayView.
              previewDisplayView.setVisibility(View.VISIBLE);
            });
    cameraHelper.startCamera(this, CameraHelper.CameraFacing.BACK, /*surfaceTexture=*/ null);
  }

}
