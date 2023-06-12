# Guide to post modifying usd attributes of newly create robots

## Import options from URDF

1. Create new scene by clicking "File" > "New from stage template" > "Empty"
1. Select only "Import Inertia Tensor" and "Create instanceable Asset"
1. Modify joint type as needed (for example, position drive, stiffness 1000)
1. "Input File" will be the URDF file of the robot
1. "Output Directory" will be the parent directory of the folder containing the usd files. (Name of the folder will be the same as the URDF file)
1. Click "Import"

## Save the instanceable model

1. After clicking import in last section, Alt+S to save the instanceable model, this is the actual model of your agent used in Isaac Sim environments.
1. Saving this model in the same directory as the "instanceable_meshes.usd" is recommended.

## Configure the collisions of the instanceable meshes

Not all the collisions are necessary and reducing the number of colliders will accelerate simulation.

1. Open "instanceable_meshes".
1. Enable "Script Editor" in "Window".
1. Copy the codes in "setup_collisions.py" and paste them into the editor.
1. Modify the names of the links whose collision will be disabled.
1. Click "Run" at the bottom of the editor.

## Optimize the collider approximation
The default approximation mathod is to transform the mesh into its corresponding convex hull. However it is not suitable for many cases. You should manually change it to other proper approximation methods such as "Convex decomposition". Taking the quadruped robot as an example:

1. For "quadruped_frame", "link1/2/3/4", select "collisions" > "mesh_0/1" under each link.
1. Change approximation and the parameters under "Physics" > "Collider".
1. Save the usd.

## Modify drive settings of the instanceable robot usd

1. Open the instanceable robot usd.
1. Copy the codes from "config_module_joints.py" to the script editor.
1. Modify the codes as needed.
1. Click "Run".
1. Do not forget to save the usd file.

## (Optional) Change the articulation root
It seems that at least for quadruped robot, you should change the articulation root from "/World/quadruped_robot" to "/World/quadruped_robot/quadruped_frame". 
