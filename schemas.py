from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class MasterPlan(BaseModel):
    concept: str
    narrative_arc: str
    visual_motifs: List[str]
    locations: List[str]
    wardrobe_progression: List[str]
    color_progression: List[str]
    performer_usage_plan: str
    section_plans: List[Dict]
    rules: List[str]

class ShotPlan(BaseModel):
    shot_id: int
    scene_id: str
    shot_type: str
    performer_visible: bool
    video_mode: str
    audio_sync_priority: str = 'high'

    transition_mode: str
    match_anchor: Optional[str] = None
    match_preserve: List[str] = []
    match_change: List[str] = []

    location: str
    action: str
    visual_metaphor: Optional[str] = None

    shot_size: str
    camera_angle: str
    camera_motion: str

    lighting: str
    palette: str

    scene_prompt: str
    motion_prompt: str

    ending_composition: str
    next_shot_hint: str

    motion_reference_id: Optional[str] = None
    motion_reference_strength: str = 'none'

    # V19 director metadata used by the exact H3 full-reference composer.
    # Defaults retain compatibility with older cached/local plans.
    duration_sec: float = 5.0
    lyrics_excerpt: str = ''
    section_role: str = 'unknown'
    performance_mode: str = 'narrative'
    wardrobe: str = 'directed scene wardrobe'
    beat_sequence: List[str] = Field(default_factory=list)


class DirectorBundle(BaseModel):
    """One-call whole-song result returned by the V19 cloud Director."""

    master_plan: MasterPlan
    shot_plans: List[ShotPlan]

class ShotSummary(BaseModel):
    shot_id: int
    scene_id: str
    location: str
    shot_size: str
    camera_motion: str
    visual_metaphor: Optional[str] = None
    transition_mode: str
    video_mode: str
    performer_visible: bool
    ending_composition: str

class GlobalUsage(BaseModel):
    locations: Dict[str, int] = {}
    camera_moves: Dict[str, int] = {}
    shot_sizes: Dict[str, int] = {}
    metaphors: Dict[str, int] = {}
    transition_modes: Dict[str, int] = {}
    video_modes: Dict[str, int] = {}

    def record(self, shot: ShotPlan):
        """Update counters from a completed shot."""
        self.locations[shot.location] = self.locations.get(shot.location, 0) + 1
        self.camera_moves[shot.camera_motion] = self.camera_moves.get(shot.camera_motion, 0) + 1
        self.shot_sizes[shot.shot_size] = self.shot_sizes.get(shot.shot_size, 0) + 1
        if shot.visual_metaphor:
            self.metaphors[shot.visual_metaphor] = self.metaphors.get(shot.visual_metaphor, 0) + 1
        self.transition_modes[shot.transition_mode] = self.transition_modes.get(shot.transition_mode, 0) + 1
        self.video_modes[shot.video_mode] = self.video_modes.get(shot.video_mode, 0) + 1

class DirectorState(BaseModel):
    master_plan: Optional[MasterPlan] = None
    current_shot: int = 0
    recent_shots: List[ShotSummary] = []
    global_usage: GlobalUsage = GlobalUsage()

    def add_shot(self, shot: ShotPlan):
        if any(item.shot_id == shot.shot_id for item in self.recent_shots):
            self.current_shot = max(self.current_shot, shot.shot_id)
            return
        summary = ShotSummary(
            shot_id=shot.shot_id,
            scene_id=shot.scene_id,
            location=shot.location,
            shot_size=shot.shot_size,
            camera_motion=shot.camera_motion,
            visual_metaphor=shot.visual_metaphor,
            transition_mode=shot.transition_mode,
            video_mode=shot.video_mode,
            performer_visible=shot.performer_visible,
            ending_composition=shot.ending_composition,
        )
        self.recent_shots.append(summary)
        if len(self.recent_shots) > 5:
            self.recent_shots = self.recent_shots[-5:]
        self.global_usage.record(shot)
        self.current_shot = shot.shot_id

def validate_semantic_rules(shot: ShotPlan) -> ShotPlan:
    if shot.shot_id <= 1:
        shot.transition_mode = 'fresh'

    if shot.transition_mode == 'continue':
        shot.video_mode = 'i2v'
    elif shot.performer_visible:
        shot.video_mode = 'multiref'
    elif shot.transition_mode == 'match_cut':
        shot.video_mode = 'i2v'
    else:
        shot.video_mode = 't2v'

    if shot.transition_mode == 'match_cut' and shot.match_anchor is None:
        shot.match_anchor = 'composition'

    if shot.motion_reference_id is None:
        shot.motion_reference_strength = 'none'

    if shot.video_mode not in ('t2v', 'i2v', 'multiref'):
        shot.video_mode = 't2v'

    if shot.transition_mode not in ('fresh', 'match_cut', 'continue'):
        shot.transition_mode = 'fresh'

    if shot.audio_sync_priority not in ('low', 'high'):
        shot.audio_sync_priority = 'high'

    return shot
