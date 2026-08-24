import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';

export type FacePose = { x: number; y: number; width: number; height: number; depth: number };
let landmarker: FaceLandmarker | null = null;

export async function setupFaceTracking(): Promise<void> {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );
  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: true,
  });
}

export function detectFace(video: HTMLVideoElement, nowMs: number): FacePose | null {
  if (!landmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return null;
  const result: FaceLandmarkerResult = landmarker.detectForVideo(video, nowMs);
  const points = result.faceLandmarks[0];
  if (!points?.length) return null;
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of points) { minX = Math.min(minX, p.x); minY = Math.min(minY, p.y); maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y); }
  const center = points[1] ?? points[0];
  return { x: center.x, y: center.y, width: maxX - minX, height: maxY - minY, depth: Math.max(0.08, 1 - center.z) };
}
