#show link: set text(fill: blue)

#let title = "[G/M]aze tracking"
#let authors = "Rongzhi Chen, Gabe Gonzalez, Rui Huang, Noah Schiro, Minh-Thy Tyler"

#set page(paper: "us-letter", margin: (x: 1in, y: 1in, top: 0.5in))
#set document(title: title, author: authors)
#set par(spacing: 0.8em)
#set enum(spacing: 0.55em)
#set list(spacing: 0.55em)
#show heading: set text(size: 11pt)
#show heading: set block(above: 0.75em, below: 0.35em)

#align(center, text(size: 14pt, weight: "bold")[#title])
#align(center, authors)

= Introduction
short problem statement here

== Background
Our project explores webcam-based eye tracking as a way to interact with a computer through gaze. We will build a software that uses a standard laptop webcam to estimate where the user is looking and translate that information into controls of an on-screen game. Our demonstration of our system will be a maze (subject to change, may explore other formats i.e. racing game) where the user can control how to solve it through eye movements.

== Motivation
We are motivated by the prospect of making the digital world more accessible to people who have difficulty using a mouse, keyboard, or handheld controller. Potential users include people with upper-body injuries, paralysis affecting limbs, or medical conditions such as arthritis that can make hand movements difficult. Using a standard webcam device–something that most people have in their homes–reduces the need for expensive and specialized hardware. Beyond accessibility, our project also has commercial applications–where users who have their hands full can interact with their laptop.
Our interest in machine learning and accessibility led us to choose a game as an engaging way to explore the technology we aim to create. The game will provide entertainment across all ages. Beyond players, product users could be caregivers, rehabilitation specialists, developers, and many others. While our semester project will focus on a playable demonstration, we hope that it will help us understand the potential for webcam-based eye tracking and how it can support other forms of hands-free device interaction. 

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

With this design, most of the heavy lifting happens client side. The primary bottleneck will be delivering the model weights when a user first connects with the app. For this reason, we likely want to use GCP Cloud Storage with a CDN attached to it. The CDN, after a cache warmup, will deliver the model weights extremely fast (assuming users are geographically centralized). Firebase hosting can serve the static (React) webpage. Realistically, this system could easily scale to _hundreds of thousands_ of concurrent users.

#figure(
  image("./assets/arch.png", width: 90%),
  caption: [
    A diagram of the system archictecture.
  ],
)

== Risks and limitations
The main challenge of our project is achieving reliable eye tracking within the time limitations of a semester. The model may perform inconsistently across users because reliable eye tracking requires individual calibration. Eye appearances vary from user to user, so a model that works well for one participant may perform less accurately for another. Examples including glasses and heavy eye makeup all create additional challenges during model calibration and use.
Another limitation is the diversity of our training data. Relying solely on MPII would limit our data to 15 participants–severely restricting our models ability to adapt to new users. To address this, we are incorporating GazeCapture, a dataset that uses mobile-device cameras. However, although adding GazeCapture would broaden our training data, it may not improve performance as there would be differences in camera hardware and viewing angles across our two datasets. We will need to evaluate how well the model transfers to webcam use and determine whether additional calibration is required.
Our approach also depends on standard webcams, which have limited frame rates compared with dedicated eye tracking hardware. Rapid eye movements may occur between frames, making it harder for our model to estimate where a user is looking. Environmental factors matter as well–changes in room lighting, shadows, and screen brightness could reduce accuracy even for a user who is already calibrated to our system. Finally, eye movement doesn’t always reflect a deliberate choice. A user may glance at an object without intending to select it, so our system will need a way to distinguish between casual looking and intentional interaction. 


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

= References
- GazeTracking dataset: Kyle Krafka, Aditya Khosla, Petr Kellnhofer, Harini Kannan, Suchi Bhandarkar, Wojciech Matusik and Antonio Torralba. “Eye Tracking for Everyone”. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. (https://gazecapture.csail.mit.edu/)
- MPIIFaceGaze: https://www.collaborative-ai.org/research/datasets/MPIIFaceGaze/
- Data normalization: Zhang et al. "Appearance-Based Gaze Estimation in the Wild". CVPR, 2015.
- Ridge regression: https://ieeexplore.ieee.org/document/11427170/
