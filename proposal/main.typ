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
#lorem(100)

== Motivation
#lorem(100)

= Data Sources
- #link("https://gazecapture.csail.mit.edu/")[GazeCapture]: Data from ~1,500 participants, contains over 2.5 million images, data collected and intended for mobile phone cameras, but will serve well as a pretraining dataset
- #link("https://www.collaborative-ai.org/research/datasets/MPIIFaceGaze/")[MPIIFaceGaze]: Slightly higher quality dataset, and uses laptop webcams. Unfortunately it only contains 213k images across 15 participants.
- Calibration data. Before participants begin the game, they will be guided through a calibration test (looking at ~9-20 points on the screen to fine tune the model further).

== Data Key Attributes
#lorem(100)

== Data Relevance
#lorem(100)

== Data Quality
#lorem(100)

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
What problems might arise?

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
- Ridge regression: https://ieeexplore.ieee.org/document/11427170/
