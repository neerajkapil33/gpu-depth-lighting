import { LIGHT_Z_MAX, LIGHT_Z_MIN, defaultRelightingSettings } from './renderer.ts';

const ORBIT_SPEED = 0.00024;
const ORBIT_RADIUS = 0.26;
const WHEEL_STEP_LIMIT = 60;
const WHEEL_SENSITIVITY = 0.0015;
const PINCH_SENSITIVITY = 0.004;
const LIGHT_GRAB_RADIUS = 0.08;
const TAP_SLOP = 0.012;
const CAMERA_ASPECT = 16 / 9;

const LightControl = { ORBIT: 'orbit', CURSOR: 'cursor', PINNED: 'pinned' } as const;
type LightControl = (typeof LightControl)[keyof typeof LightControl];
type Gesture =
  | { readonly kind: 'none' }
  | { readonly kind: 'press'; readonly grabbed: boolean; readonly x: number; readonly y: number }
  | { readonly kind: 'drag' }
  | { kind: 'pinch'; span: number };

interface LightUpdate { lightPosition?: [number, number]; lightZ?: number }
export interface LightInput { readonly lightPosition: [number, number]; readonly lightZ: number; orbitTick(): void }

const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));

/** Natural screen-space light control: idle orbit, hover steering, click/drag pinning, wheel/pinch depth. */
export function setupLightInput(canvas: HTMLCanvasElement, onChange: (update: LightUpdate) => void, signal: AbortSignal): LightInput {
  const pointers = new Map<number, { x: number; y: number }>();
  let gesture: Gesture = { kind: 'none' };
  let control: LightControl = LightControl.ORBIT;
  let lightPosition: [number, number] = [...defaultRelightingSettings.lightPosition];
  let lightZ = defaultRelightingSettings.lightZ;
  const mobileTouch = () => window.matchMedia('(max-width: 850px)').matches;

  const placeLight = (x: number, y: number) => { lightPosition = [clamp(x, 0, 1), clamp(y, 0, 1)]; onChange({ lightPosition }); };
  const pushLight = (amount: number) => { lightZ = clamp(lightZ + amount, LIGHT_Z_MIN, LIGHT_Z_MAX); onChange({ lightZ }); };
  const pinLight = (pinned: boolean) => { control = pinned ? LightControl.PINNED : LightControl.CURSOR; };

  // On mobile the rendered canvas uses object-fit: contain. Map touch coordinates
  // into the actual 16:9 image area so taps/drags do not include letterbox bars.
  const fraction = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    if (!r.width || !r.height) return undefined;
    if (e.pointerType !== 'touch' || !mobileTouch()) return { x: (e.clientX-r.left)/r.width, y: (e.clientY-r.top)/r.height };
    const displayedWidth = Math.min(r.width, r.height * CAMERA_ASPECT);
    const displayedHeight = displayedWidth / CAMERA_ASPECT;
    const left = r.left + (r.width - displayedWidth) / 2;
    const top = r.top + (r.height - displayedHeight) / 2;
    return { x: (e.clientX-left)/displayedWidth, y: (e.clientY-top)/displayedHeight };
  };
  const overLight = (p: {x:number;y:number}) => Math.hypot(p.x-lightPosition[0], p.y-lightPosition[1]) <= LIGHT_GRAB_RADIUS;
  const span = () => { const [a,b] = [...pointers.values()]; return a&&b ? Math.hypot(a.x-b.x,a.y-b.y) : 0; };

  function down(e: PointerEvent) {
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});
    if (pointers.size >= 2) { gesture={kind:'pinch',span:span()}; return; }
    const p=fraction(e); if(!p)return;
    const grabbed=overLight(p); gesture={kind:'press',grabbed,x:p.x,y:p.y};
    if(!grabbed && e.pointerType!=='touch'){placeLight(p.x,p.y);pinLight(true);}
  }
  function move(e: PointerEvent) {
    if(pointers.has(e.pointerId)) pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});
    if(gesture.kind==='pinch'){e.preventDefault();const s=span();if(gesture.span>0)pushLight((s-gesture.span)*PINCH_SENSITIVITY);gesture.span=s;return;}
    const p=fraction(e);if(!p)return;
    if(gesture.kind==='press'&&Math.hypot(p.x-gesture.x,p.y-gesture.y)>TAP_SLOP){gesture={kind:'drag'};pinLight(true);placeLight(p.x,p.y);}
    else if(gesture.kind==='drag')placeLight(p.x,p.y);
    else if(gesture.kind==='none'){canvas.style.cursor=overLight(p)?'grab':'crosshair';if(control===LightControl.CURSOR)placeLight(p.x,p.y);}
  }
  function up(e: PointerEvent) {
    pointers.delete(e.pointerId);if(gesture.kind==='pinch'){gesture.span=span();}
    if(pointers.size)return;
    if(gesture.kind==='press'&&e.type==='pointerup'){if(gesture.grabbed)pinLight(control!==LightControl.PINNED);else if(e.pointerType==='touch'){placeLight(gesture.x,gesture.y);pinLight(true);}}
    gesture={kind:'none'};
  }
  function enter(){if(control!==LightControl.PINNED)control=LightControl.CURSOR;}
  function leave(){if(control!==LightControl.PINNED)control=LightControl.ORBIT;canvas.style.cursor='crosshair';}
  function wheel(e: WheelEvent){e.preventDefault();let d=e.deltaY;if(e.deltaMode===WheelEvent.DOM_DELTA_LINE)d*=16;else if(e.deltaMode===WheelEvent.DOM_DELTA_PAGE)d*=canvas.clientHeight;d=Math.sign(d)*Math.min(Math.abs(d),WHEEL_STEP_LIMIT);pushLight(d*WHEEL_SENSITIVITY);}

  canvas.addEventListener('pointerdown',down,{signal});canvas.addEventListener('pointermove',move,{signal});canvas.addEventListener('pointerup',up,{signal});canvas.addEventListener('pointercancel',up,{signal});canvas.addEventListener('pointerenter',enter,{signal});canvas.addEventListener('pointerleave',leave,{signal});canvas.addEventListener('wheel',wheel,{passive:false,signal});
  return {get lightPosition(){return lightPosition},get lightZ(){return lightZ},orbitTick(){if(control!==LightControl.ORBIT)return;const phase=performance.now()*ORBIT_SPEED;placeLight(.5+Math.cos(phase)*ORBIT_RADIUS,.44+Math.sin(phase*1.37)*ORBIT_RADIUS*.8);}};
}
