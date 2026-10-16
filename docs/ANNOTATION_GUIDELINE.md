# Surgical Annotation Guideline v1.0

Official guideline for annotating laparoscopic surgical video frames, anatomical structures, and surgical tools.

## 1. Quality & Agreement Thresholds
- **Inter-rater Agreement:** Fleiss kappa >= 0.70, mean pairwise Dice >= 0.80.
- **Automation Bias Safeguards:**
  - 10% of frames presented without pre-annotation (blind control).
  - Target delta-contour Dice >= 0.90 between pre-annotated and scratch annotations.
  - Acceptance rate of SAM 2 suggestions must stay < 60%.

## 2. Anatomical Classes & Rules
1. **Liver:** Include parenchyma; exclude gallbladder bed when dissected.
2. **Gallbladder:** Include fundus, body, neck; stop at cystic duct junction.
3. **Hepatocystic Triangle / Calot's Triangle:** Critical view of safety landmarks.
4. **Surgical Instruments:** Shaft, joint, and active jaws are annotated with sub-part labels.

## 3. Disputed Boundaries & Adjudications
In cases of severe inflammation or anatomical distortion, mark as `needs-clinician` for review in `docs/adjudications.md`.

## 4. Annotating Video in CVAT

This section describes the tooling workflow only. It does not change class definitions (Section 2), agreement thresholds (Section 1), or the adjudication process (Section 3). CVAT menus and options change between releases, so check the [official CVAT documentation](https://docs.cvat.ai/docs/) for the version deployed in your project.

### 4.1 Before you start
- Work only with de-identified videos that have passed the OSV de-identification pipeline. Never upload raw or burned-in-text footage to a CVAT instance.
- Never commit videos, exported frames, or any other raw images to git. Data lives in DVC only; exports from CVAT are annotation files.
- Use the project's label set as provided. Do not add labels or change label attributes on your own; ask the maintainers instead.

### 4.2 Task setup (maintainers)
Tasks are created from the **Tasks** page: **+** > **Create new task**. For a video, the following advanced options matter ([Tasks](https://docs.cvat.ai/docs/workspace/tasks-page/)):

| Option | What it does |
|---|---|
| Segment size | Splits the video into parts, each of which becomes a separate job. |
| Start frame / Stop frame | First and last frame to include. |
| Frame step | Includes every Nth frame (step 25 on a 100-frame video gives frames 1, 26, 51, 76, 100). |
| Overlap size | Overlapping segments for continuous tracking; works for bounding boxes only. |
| Chunk size, Image quality | Streaming and quality settings. Raise image quality only if small objects are not visible. |
| Prefer zip chunks | Recommended off for video. |
| Sorting method | Not applicable to video. |

Labels and attributes are defined on the **Constructor** tab. A **mutable** attribute can change between frames of a track (for example, a state of an instrument), an **immutable** one cannot. Attribute types are Select, Radio, Checkbox, Text, and Number.

Use the segment size, frame step, and other values given in the task description. If none are given, ask the maintainers before annotating.

### 4.3 Annotating with tracks
Use **track mode** for any object that persists across frames. Tracks store one object identity over time and interpolate its shape between keyframes ([Track mode](https://docs.cvat.ai/docs/annotation/manual-annotation/modes/track-mode-basics/)).

1. Select a drawing tool in the sidebar, choose the label, and select **Track** (not Shape) in the drawing form. Draw the object on the first frame where it is visible.
2. Move to a later frame where the object has moved or changed, correct the shape, and mark that frame as a keyframe (star button or `K`). Frames between keyframes are interpolated.
3. Scrub through the interpolated frames and add keyframes wherever the interpolated shape drifts off the object. Fast-moving instruments and deforming tissue need more keyframes than static objects.
4. When the object leaves the field of view or can no longer be seen, apply the **Outside** property (`O`) so the track ends there.
5. If the object comes back, create a new track for the new period and join the pieces with **Merge** (`M`, select a shape on each track, press `M` again), so the object keeps one identity. **Split** divides one track in two.
6. Use **Occluded** (`Q`) for an object that is present but partly hidden, for example by an instrument or smoke. This is different from Outside, which means the object is absent.

Other useful controls ([Objects sidebar](https://docs.cvat.ai/docs/annotation/annotation-editor/objects-sidebar/)): **Lock** (`L`), **Propagate** (`Ctrl+B`, copies an object to the next or previous N frames at the same position), and the frame jump buttons (`C` and `V` move back and forward by the Player Step, 10 frames by default). Full lists are in **Settings > Shortcuts** ([Shortcuts](https://docs.cvat.ai/docs/getting_started/shortcuts/)).

For single-point objects and attribute-only labelling, see [Linear interpolation with one point](https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/annotation-with-points/liner-interpolation-with-one-point/) and [Attribute annotation mode](https://docs.cvat.ai/docs/annotation/manual-annotation/modes/attribute-annotation-mode-basics/).

### 4.4 SAM 2 tracking in CVAT
CVAT offers a SAM 2 Tracker that follows an existing polygon or mask forward through the video ([SAM 2 Tracker](https://docs.cvat.ai/docs/annotation/auto-annotation/segment-anything-2-tracker/)). Rectangles and skeletons are not supported.

- Draw or select a polygon or mask, then **Run annotation action** and choose the SAM 2 tracker, and set the frame where tracking should stop.
- If the tracked shape degrades, correct it and run the tracker again from that frame.
- It is available only in CVAT Online and Enterprise (the agent-based version needs CVAT 2.42.0 or later) and runs on a deployed agent; a GPU is recommended.
- Tracker output is a SAM 2 suggestion. Review every frame, and the pre-annotation safeguards in Section 1 (blind control frames and the acceptance-rate limit) apply to it as well.

### 4.5 Job state and review
Each job has a **stage** (annotation, validation, acceptance) and a **state** (new, in progress, rejected, completed). The assignee changes the state from the job **Menu** ([Menu and navigation bar](https://docs.cvat.ai/docs/annotation/annotation-editor/navbar/)). Save regularly with the **Save** button, and set the state to completed only when every object in your job is annotated. Roles and the review workflow are described in the [organization workflow guide](https://docs.cvat.ai/docs/guides/workflow-org/).

CVAT's built-in [automated quality control](https://docs.cvat.ai/docs/qa-analytics/auto-qa/) (CVAT Online and Enterprise) compares jobs with a Ground Truth job, typically 5-15% of the frames. For video, use a Ground Truth job; honeypots do not support tracks. This report is a supplement, and it does not replace the agreement metrics in Section 1.

### 4.6 Export
- Task: **Actions** > **Export task dataset**. Job: **Menu** > **Export job dataset**. Pick the format, name the archive, and click **OK** ([Export datasets](https://docs.cvat.ai/docs/dataset_management/export-datasets/)). Leave **Save images** off.
- **CVAT for video 1.1** keeps tracks with their `keyframe`, `outside`, and `occluded` flags and per-frame attributes ([format description](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/)). Deliver the format named in the task description.
- **MOTS PNG** exports only tracks, as per-frame instance masks ([MOTS](https://docs.cvat.ai/docs/dataset_management/formats/format-mots/)). Shapes that are not part of a track are dropped, so make sure every object is a track before exporting to it.

### 4.7 Further reading
- [CVAT documentation](https://docs.cvat.ai/docs/)
- [Annotation editor](https://docs.cvat.ai/docs/annotation/annotation-editor/)
- [Manual annotation](https://docs.cvat.ai/docs/annotation/manual-annotation/)
- [Import datasets](https://docs.cvat.ai/docs/dataset_management/import-datasets/) and [export formats](https://docs.cvat.ai/docs/dataset_management/formats/)
- [Vocabulary](https://docs.cvat.ai/docs/getting_started/vocabulary/)
