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

= Problem Statement
People who cannot use a mouse or keyboard have few good options: dedicated eye-tracking hardware is a separate, specialized purchase, and voice control is unusable in noisy spaces. We will develop an application that turns the laptop's built-in webcam into a gaze-driven pointer for two-dimensional interfaces and demonstrate its efficacy by steering a ball through a maze.

= Background and Motivation
Appearance-based gaze estimation infers where a person is looking from ordinary camera images, without the infrared hardware eye trackers require. We use a standard laptop webcam to estimate the user's gaze point and turn it into a hands-free control signal.

Our stakeholders are people for whom a mouse and keyboard are difficult or impossible, including those with limb paralysis, arthritis, injury, or those who cannot afford additional eye-tracking equipment. A laptop webcam is already present, so the remaining barrier is software rather than equipment. Additionally, we target anyone whose hands are occupied: reading a recipe while cooking, following instructions while assembling parts, consulting a protocol in a gloved wet lab. Voice is the other hands-free channel, but it fails in shared or noisy rooms and excludes people with speech impairments; gaze is silent.

The same interaction model is moving into wearables. Apple Vision Pro already makes gaze the primary pointer, so what we learn transfers beyond the laptop, even though headsets sense gaze with infrared hardware. We chose a game as the demonstration because a control task makes real-time behavior visible in a way an offline accuracy number does not.

= Data Sources
- #link("https://gazecapture.csail.mit.edu/")[GazeCapture]: Data from ~1,500 participants, contains over 2.5 million images, data collected and intended for mobile phone cameras, but will serve well as a pretraining dataset
- #link("https://www.collaborative-ai.org/research/datasets/MPIIFaceGaze/")[MPIIFaceGaze]: Slightly higher-quality dataset, and uses laptop webcams. Unfortunately it only contains 37k images across 15 participants.
- Calibration data. Before participants begin the game, they will be guided through a calibration test (looking at ~9-20 points on the screen to fine-tune the model further).

== Data Key Attributes
MPIIFaceGaze ships one annotation file per participant (`pXX.txt`, one row per image, 28 columns) plus a `Calibration` folder with `Camera.mat` (intrinsics, distortion), `monitorPose.mat` (screen plane pose in camera coordinates) and `screenSize.mat` (screen size in pixels and mm). We use:
- *Image path* (col. 1): the full-face frame, from which a landmark detector (identical in training and in the browser) yields face and eye crops.
- *Gaze target on screen* (cols. 2-3), in pixels. Divided by the `screenSize.mat` pixel dimensions, this becomes our label $(x, y) in [0,1]^2$, the same space the maze runs in.
- *Facial landmarks* (cols. 4-15): four eye corners and two mouth corners, used to validate our detector and define eye crops.
- *Head pose* (cols. 16-21) and *face center* (cols. 22-24) in camera coordinates, with `Camera.mat`: normalization of crops into a canonical camera space so cameras, distances and head angles are comparable.
- *3D gaze target* (cols. 25-27): minus the face center gives gaze direction. This can be an alternative label if we predict direction and intersect the screen plane from `monitorPose.mat`.
- *Participant ID* (folder): for person-level train/validation splits. Col. 28 (evaluation eye) is not needed.

GazeCapture provides analogous JSON fields (dot position, orientation, screen size, face/eye boxes). Our *calibration frames* (9 to 20 per player) pair an image with a known screen-fraction target, for on-device ridge regression only.

== Data Relevance
The task is appearance-based gaze estimation: predict where on the screen a person is looking from one webcam frame, then use that to control the game. MPIIFaceGaze matches our deployment setting: laptop webcams over months of everyday use, covering lighting, distance, head-pose and glasses variation. Its labels are already screen points and it ships the camera and screen geometry needed for normalization, so no extra annotation is needed. GazeCapture is mobile, but has two orders of magnitude more people. This dataset may be needed to generalize across identities. We may pretrain on GazeCapture, then fine-tune and evaluate on MPIIFaceGaze in screen-fraction error.

== Data Quality
- *Few identities.* MPIIFaceGaze has 37k images but 15 people, so a model can memorize faces. Preliminary plan: leave-one-person-out splits and per-participant error reporting.
- *Domain gap.* Phone vs. laptop cameras and geometry. Preliminary plan: GazeCapture for pretraining only, validate on laptop data.
- *Landmark mismatch.* Shipped landmarks and head pose come from a 6-point model, not our browser detector. Preliminary plan: recompute landmarks with our detector.
- *Calibration variability.* Players may move or look away. Preliminary plan: require a minimum of valid frames per crosshair, allow redo.

= Scope and Objectives
The project will have a few primary components:
- Model training
  - During initial development and setup of the training pipeline, most development can happen locally (on small consumer GPUs).
  - For the final model training, we will likely want to utilize cloud compute resources. This can greatly speed up the search for the best hyperparameters.
- Model inference
  - Many modern gaze-tracking models are small enough to run in the browser. Even for the fine-tuning step, algorithms commonly use #link("https://en.wikipedia.org/wiki/Ridge_regression")[ridge regression] which can be computed within milliseconds.
  - At inference, cloud computing will primarily be responsible for serving the webpage and delivering the model weights.

Most of the heavy lifting happens client-side, so the primary bottleneck is delivering the model weights when a user first connects. We plan to serve them from GCP Cloud Storage behind a CDN. The CDN ensures that we have extremely low load times when a user first connects to the website. Firebase will host the static React page. This system can scale realistically scale to _hundreds of thousands_ of concurrent users.

Inference has real computational constraints we must work around. The model will be running in the browser and needs to be small enough to be fine tuned on CPU and run inference several times a second. However, the model must also be large enough that it can reliably estimate gaze.

= Learning Emphasis
The project emphasizes convolutional neural networks for appearance-based gaze regression, transfer learning across two datasets with different camera geometry, and ridge regression for per-user calibration, alongside the containerization, experiment tracking and cloud deployment practices covered in the course.

= Application Mock Design
- *User Interface 1:* Calibration screen. The user is guided through a set of on-screen points; a frame is captured at each one and used to fit the per-user model.
- *User Interface 2:* Maze screen. The user steers a ball to the exit using gaze alone, with a win condition and a completion time.

#figure(
  image("./assets/arch.png", width: 70%),
  caption: [
    A diagram of the system architecture.
  ],
)

= Limitations and Risks
- *Per-user variation.* Eye appearance differs enough between people that a model calibrated for one can be noticeably worse for another. Glasses and heavy eye makeup make this worse.
- *Narrow training data.* MPIIFaceGaze covers 15 people. GazeCapture adds identity diversity but was shot on phone cameras, so transfer to laptop webcams is unproven and may need more calibration than we have budgeted.
- *Commodity hardware.* Webcams run at low frame rates, so fast eye movements fall between frames. Lighting, shadows and screen brightness degrade accuracy even after a user is calibrated.
- *Looking is not commanding.* Eyes explore as much as they act, so the system cannot treat every glance as input. This is the Midas touch problem. Dwell-time confirmation is a standard fix, but applies to discrete selection rather than continuous steering, which makes it the risk most specific to the maze.
- *Fallback if precision falls short.* If gaze proves too coarse to steer with, we fall back to a gaze cursor for browsing: scroll by gaze position, or select by dwell. Targets can be made larger and errors are self-correcting. The model, datasets and deployment path are unchanged, so only the UI layer differs.

= Fun Factor
Demonstrating the gaze tracking system as a means of controlling a game is both engaging, fun, and can get people excited about the intersection between machine learning and accessibility.

= Milestones
The #link("https://github.com/NoahSchiro/ac215/")[repository] is currently set in such a way that CI is enforced, and a minimum level of code coverage is required for PRs to be merged. Milestones related to CI / code coverage were already met at the beginning of the project and will be maintained as such.

== Milestone 2 (10/20)
- Maze generation and display in Web UI; player controlled with arrow keys. Game has an end state / win condition.
- Dataset ingestion pipeline is established.
- Model architecture finalized.
- Full pretraining and evaluation pipeline established.
- Hyperparameter tuning guided via experiment tracking software (e.g. MLflow, W&B)

== Milestone 3 (11/12)
- Calibration screen is created, webpage takes images at each calibration checkpoint and stores them.
- Model is able to perform ridge regression on the stored calibration images.
- Maze controls switched to gaze.

== Milestone 4 (12/1)
- Web UI is containerized.
- Webpage is hosted on Firebase.
- Model weights are stored in GCP Cloud Storage and connected to the CDN and the webpage.

== Milestone 5 (12/11)
- Continuous deployment (on merge) is established.
- Deployment is switched to Kubernetes for easy scaling.
- Final presentation is prepared.

= Research and Development
- GazeCapture: Krafka et al., #link("https://gazecapture.csail.mit.edu/")[“Eye Tracking for Everyone”]. CVPR 2016.
- MPIIFaceGaze dataset: Zhang et al., #link("https://openaccess.thecvf.com/content_cvpr_2017_workshops/w41/papers/Bulling_Its_Written_All_CVPR_2017_paper.pdf")["It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation"]. CVPRW 2017.
- Data normalization: Zhang et al., #link("https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation")["Revisiting Data Normalization for Appearance-Based Gaze Estimation"]. ETRA 2018.
- Browser deployment: Papoutsaki et al., #link("https://www.ijcai.org/Proceedings/16/Papers/540.pdf")["WebGazer: Scalable Webcam Eye Tracking Using User Interactions"]. IJCAI 2016.
- Evaluation: Zhang et al., #link("https://arxiv.org/abs/1901.10906")["Evaluation of Appearance-Based Methods and Implications for Gaze-Based Applications"]. CHI 2019.
