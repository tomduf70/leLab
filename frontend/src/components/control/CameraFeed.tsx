import React, { useState } from "react";
import { VideoOff } from "lucide-react";
import { useCameraStream } from "@/hooks/useCameraStream";
import { useApi } from "@/contexts/ApiContext";

interface CameraFeedProps {
  /** Browser deviceId to stream via getUserMedia. Empty ⇒ try the backend stream. */
  deviceId: string;
  /**
   * cv2 camera index for the backend MJPEG fallback, used when the browser has no
   * matching device (remote access — see the /camera-stream endpoint). Undefined
   * ⇒ no fallback, just the placeholder.
   */
  cameraIndex?: number;
  /** Optional caption shown under the feed. */
  label?: string;
}

/**
 * Live camera feed. Prefers the browser's own camera via getUserMedia (fast, local);
 * when there's no matching browser device (remote access over lelab.iscol.fr) it
 * falls back to the host's backend MJPEG stream so the feed still works through the
 * tunnel.
 */
const CameraFeed: React.FC<CameraFeedProps> = ({ deviceId, cameraIndex, label }) => {
  const { baseUrl } = useApi();
  const { videoRef, hasError } = useCameraStream(deviceId, false);
  const [streamFailed, setStreamFailed] = useState(false);

  const showVideo = deviceId && !hasError;
  const showStream = !showVideo && cameraIndex != null && !streamFailed;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
      <div className="aspect-[4/3] bg-gray-800 relative">
        {showVideo ? (
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover"
          />
        ) : showStream ? (
          <img
            src={`${baseUrl}/camera-stream/${cameraIndex}`}
            alt={label ?? "Camera stream"}
            onError={() => setStreamFailed(true)}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center">
            <VideoOff className="w-8 h-8 text-gray-500 mb-2" />
            <span className="text-gray-500 text-sm">
              {deviceId ? "Preview failed" : "No camera selected"}
            </span>
          </div>
        )}
      </div>
      {label && (
        <div className="p-2 text-sm text-gray-300 truncate border-t border-gray-800">
          {label}
        </div>
      )}
    </div>
  );
};

export default CameraFeed;
