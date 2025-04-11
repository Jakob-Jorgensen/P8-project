## P8 Project - MAX VLM

**Welcome to our two-device project.**
In this branch, you will find your code executed on a Jetson AGX Orin, which we call Max.

### Functionality  
Max is responsible for detecting objects with vision, gathering information about the environment, and then providing instructions to our Franka robot based on verbal commands from the user.

### Components  
- **VLM + GroundingDino, Lama 0.5B**: Combined with GroundingDino, this gives bounding boxes of a singular desired object.  
- **gg-CNN (Grasping)**: Uses the bounding box to isolate the object and generate the best grasping point.  
- **Voice Script**: Enables communication between the user and Max.
