#show link: set text(fill: blue)

#let title = "[G/M]aze tracking"
#let authors = "Rongzhi Chen, Gabe Gonzalez, Rui Huang, Noah Schiro, Minh-Thy Tyler"

#set page(paper: "us-letter", margin: (x: 0.85in, top: 0.5in, bottom: 0.7in))
#set document(title: title, author: authors)
#set text(size: 10.5pt)
#set par(spacing: 0.7em)
#set enum(spacing: 0.4em)
#set list(spacing: 0.4em)
#show heading: set text(size: 11pt)
#show heading: set block(above: 0.5em, below: 0.25em)

#align(center, text(size: 14pt, weight: "bold")[#title])
#align(center, authors)

= Introduction
Develop an application that lets a user control a two-dimensional interface using only gaze from a standard laptop webcam, demonstrated through a maze in which a ball is steered by eye movements alone. If accuracy proves too coarse for steering, we fall back to a gaze cursor — scroll by gaze, select by dwell — reusing the same model and pipeline.

== Background and Motivation
Appearance-based gaze estimation infers where a person is looking from ordinary camera images, without the infrared hardware dedicated eye trackers require. We use a standard laptop webcam to estimate the user's gaze point and turn it into a hands-free control signal.

Our stakeholders are, first, people for whom a mouse and keyboard are difficult or impossible — limb paralysis, arthritis, injury — and for whom dedicated gaze hardware is expensive and specialized; commodity webcams are already in most homes, so the barrier is software rather than equipment. Second, anyone whose hands are occupied: reading a recipe while cooking, following instructions while assembling parts, consulting a protocol in a gloved wet lab. Both groups need to read and browse rather than steer, which tolerates far more error. Voice is the other hands-free channel, but it fails in shared or noisy rooms and excludes people with speech impairments; gaze is silent.

The same interaction model is moving into wearables — Apple Vision Pro already makes gaze the primary pointer — so what we learn transfers beyond the laptop, though headsets sense gaze with dedicated infrared hardware. We chose a game as the demonstration because a control task makes real-time behaviour visible in a way an offline accuracy number does not.

= Data Sources
- #link("https://gazecapture.csail.mit.edu/")[GazeCapture]: Data from ~1,500 participants, contains over 2.5 million images, data collected and intended for mobile phone cameras, but will serve well as a pretraining dataset
- #link("https://www.collaborative-ai.org/research/datasets/MPIIFaceGaze/")[MPIIFaceGaze]: Slightly higher quality dataset, and uses laptop webcams. Unfortunately it only contains 213k images across 15 participants.
- Calibration data. Before participants begin the game, they will be guided through a calibration test (looking at ~9-20 points on the screen to fine tune the model further).

== Data Key Attributes
MPIIFaceGaze ships one annotation file per participant (`pXX.txt`, one row per image, 28 columns) plus a `Calibration` folder with `Camera.mat` (intrinsics, distortion), `monitorPose.mat` (screen plane pose in camera coordinates) and `screenSize.mat` (screen size in pixels and mm). We use:
- *Image path* (col. 1): the full-face frame, from which a landmark detector (identical in training and in the browser) yields face and eye crops.
- *Gaze target on screen* (cols. 2-3), in pixels. Divided by the `screenSize.mat` pixel dimensions this becomes our label $(x, y) in [0,1]^2$, the same space the maze runs in.
- *Facial landmarks* (cols. 4-15): four eye corners and two mouth corners, used to validate our detector and define eye crops.
- *Head pose* (cols. 16-21) and *face center* (cols. 22-24) in camera coordinates, with `Camera.mat`: drive normalization of crops into a canonical camera space so cameras, distances and head angles are comparable.
- *3D gaze target* (cols. 25-27): minus the face center gives gaze direction, an alternative label if we predict direction and intersect the screen plane from `monitorPose.mat`.
- *Participant ID* (folder): for person-level train/validation splits. Col. 28 (evaluation eye) is not needed.

GazeCapture provides analogous JSON fields (dot position, orientation, screen size, face/eye boxes). Our *calibration frames* (9 to 20 per player) pair an image with a known screen-fraction target, for on-device ridge regression only.

== Data Relevance
The task is appearance-based gaze estimation: predict where on the screen a person is looking from one webcam frame, then use that as a game controller. MPIIFaceGaze matches our deployment setting: laptop webcams over months of everyday use, covering the lighting, distance, head-pose and glasses variation of someone playing at a laptop. Its labels are already screen points and it ships the camera and screen geometry needed for normalization, so no extra annotation is needed. GazeCapture is mobile, but has two orders of magnitude more people, which a CNN needs to generalize across identities. We pretrain on GazeCapture, then fine-tune and evaluate on MPIIFaceGaze in screen-fraction error.

== Data Quality
- *Few identities.* MPIIFaceGaze has 213k images but 15 people, so a model can memorize faces. Preliminary plan: leave-one-person-out splits and per-participant error reporting.
- *Domain gap.* Phone vs. laptop cameras and geometry. Preliminary plan: GazeCapture for pretraining only, validate on laptop data.
- *Landmark mismatch.* Shipped landmarks and head pose come from a 6-point model, not our browser detector. Preliminary plan: recompute landmarks with our detector.
- *Calibration variability.* Players may move or look away. Preliminary plan: require a minimum of valid frames per cross-hair, allow redo.

= Scope and Design
The project will have a few primary components:
- A web UI
  - A calibration screen and workflow, wherein users will be instructed to look at a set of points on the screen. At each point, a photo will be taken and stored to further calibrate the model to that specific user.
  - A screen where a user can solve a maze, purely by using gaze.
- Model training
  - During initial development and setup of the training pipeline, most development can happen locally (on small consumer GPUs).
  - For the final model training, we will likely want to utilize cloud compute resources. This can greatly speed up the search for the best hyperparameters.
- Model inference
  - Many modern gaze tracking models are small enough to run in the browser. Even for the fine-tuning step, many algorithms use #link("https://en.wikipedia.org/wiki/Ridge_regression")[ridge regression] which can be computed within milliseconds.
  - At inference, cloud computing will primarily be responsible for serving the webpage and delivering the model weights.

Most of the heavy lifting happens client side, so the primary bottleneck is delivering the model weights when a user first connects. We plan to serve them from GCP Cloud Storage behind a CDN, which after a cache warmup delivers them quickly provided users are geographically clustered, with Firebase hosting the static React page. Realistically this scales to _hundreds of thousands_ of concurrent users.

#figure(
  image("./assets/arch.png", width: 90%),
  caption: [
    A diagram of the system archictecture.
  ],
)

== Risks and limitations
- *Per-user variation.* Eye appearance differs enough between people that a model calibrated for one can be noticeably worse for another. Glasses and heavy eye makeup make this worse.
- *Narrow training data.* MPIIFaceGaze covers 15 people. GazeCapture adds identity diversity but was shot on phone cameras, so transfer to laptop webcams is unproven and may need more calibration than we have budgeted.
- *Commodity hardware.* Webcams run at low frame rates, so fast eye movements fall between frames. Lighting, shadows and screen brightness degrade accuracy even after a user is calibrated.
- *Looking is not commanding.* Eyes explore as much as they act, so the system cannot treat every glance as input — the Midas touch problem. Dwell-time confirmation is the standard fix, but it only applies to discrete selection, not to continuous steering. This is the risk most specific to the maze.

= Milestones
The #link("https://github.com/NoahSchiro/ac215/")[repository] is already setup in such a way that CI is enforced, and a minimum code coverage is required for PRs to be merged. Milestones related to CI / code coverage are already met at the beginning of the project and will be maintained as such.

== Milestone 2 (10/20)
- Maze generation and display in Web UI; player controlled with arrow keys. Game has an end state / win condition.
- Dataset ingestion pipeline is established.
- Model architecture finalized.
- Full pretraining and evaluation pipeline established.
- Hyperparameter tuning guided via experiment tracking software (i.e. MLFlow, W&B, etc.)

== Milestone 3 (11/12)
- Calibration screen is setup, webpage takes images at each calibration checkpoint and stores them.
- Model is able to perform ridge regression on the stored calibration images.
- Maze controls switched to gaze.

== Milestone 4 (12/1)
- Web UI is containerized.
- Webpage is hosted on Firebase.
- Model weights are stored in GCP Cloud Storage, connected to CDN and webpage.

== Milestone 5 (12/11)
- Continuous deployment (on merge) is established.
- Deployment is switched to kubernetes for easy scale up.
- Final presentation is prepared.

= Research and Development
- GazeCapture: Krafka et al., #link("https://gazecapture.csail.mit.edu/")[“Eye Tracking for Everyone”]. CVPR 2016.
- MPIIFaceGaze dataset: Zhang et al., #link("https://openaccess.thecvf.com/content_cvpr_2017_workshops/w41/papers/Bulling_Its_Written_All_CVPR_2017_paper.pdf")["It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation"]. CVPRW 2017.
- Data normalization: Zhang et al., #link("https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation")["Revisiting Data Normalization for Appearance-Based Gaze Estimation"]. ETRA 2018.
- Browser deployment: Papoutsaki et al., #link("https://www.ijcai.org/Proceedings/16/Papers/540.pdf")["WebGazer: Scalable Webcam Eye Tracking Using User Interactions"]. IJCAI 2016.
- Evaluation: Zhang et al., #link("https://arxiv.org/abs/1901.10906")["Evaluation of Appearance-Based Methods and Implications for Gaze-Based Applications"]. CHI 2019.
