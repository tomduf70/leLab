/**
 * URL of the backend MJPEG preview for a camera. `stableId` (Linux udev
 * /dev/v4l/by-id path) wins over `index` on the backend, so the preview opens
 * the right camera even if cv2 indices shifted since the config was saved.
 */
export const cameraStreamUrl = (
  baseUrl: string,
  index: number | undefined,
  stableId?: string,
): string =>
  `${baseUrl}/camera-stream/${index}` +
  (stableId ? `?stable_id=${encodeURIComponent(stableId)}` : "");
