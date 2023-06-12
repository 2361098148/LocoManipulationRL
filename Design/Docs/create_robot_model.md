[asset]: /OmniverseGym/Docs/assets

# How to create your own robot in Omniverse Isaac Sim

## Prepare your URDF files
See [Create Robot URDF] (TODO)

## Import URDF model in Omniverse Isaac Sim
1. Open Isaac Sim; Go to Isaac Utils -> Workflows -> URDF Importer
![urdf_importer](/OmniverseGym/Docs/assets/urdf_importer_location.PNG)

1. Config import options
    - Check "Import Inertia Tensor" if you want the inertia properties in your URDF file to be inherited.
    - To ensure that your robot works normally as you expect, we suggest you disable "Self Collision"; you can config self-collision of the robot later on.
    - "Input File" should be the directory leading to your URDF file
![urdf_options](/OmniverseGym/Docs/assets/urdf_importer_options.PNG)

## Make closed-chain mechanism simulated normally
Nvidia Physx, the backend physics engine of Omniverse, uses Articulation to control the joints of the robots. Articulation does not support a close loop structure, however, we can make our closed-chain work by excluding the tip joint from the articulation tree, as explained in [Articulations ‒ Omniverse Extensions documentation](https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics/articulations.html):

>While articulations natively only support tree-structures, it is 
possible to create loops in the articulation by adding joints between 
articulation links that have the Exclude from Articulation flag enabled. For example, we could tie the ragdoll’s hands together by adding a distance joint between the two hand spheres.

Isaac Sim does not expect our robot to have a close loop structure, so we have to manually change the configurations. The process is rather tedious...but it will be worth it!

1. Exclude tip joint from articulation. For every module, there is a revolute joint called "\<prefix\>_closed_chain_revolute" under "\<prefix\>_link3_intermediate".  In the "Property" panel below, check "Exclude From Articulation".
![tip_joint](/OmniverseGym/Docs/assets/tip_joint_location.PNG) 
![closed_chain_joint](/OmniverseGym/Docs/assets/close_chain_joint_name.PNG) 
![exclude_check_box](/OmniverseGym/Docs/assets/check_exclude.PNG)

1. Delete "Drive" element for joint "\<prefix\>_closed_chain_revolute" under "\<prefix\>_link3_intermediate", "\<prefix\>_link1_to_link2" under link "\<prefix\>_link1"  as well as "\<prefix\>_link4_to_link3" under "\<prefix\>_link4";
![link4_to_link3](/OmniverseGym/Docs/assets/link4_to_link3.PNG)
![delete_joint_drive](/OmniverseGym/Docs/assets/delete_joint_drive.PNG)

1. Convex decomposition of colliders (TODO)
1. DoF parameters. Drivable DoFs are named "\<prefix\>_dof\<dof_index\>" under "\<prefix\>_xm430_0" or "\<prefix\>_xm430_2"; You can change the joint limits, maximum force or other parameters in the property penal.

![dof_names](/OmniverseGym/Docs/assets/dof_joint_names.PNG)
![dof_properties](/OmniverseGym/Docs/assets/drive_properties.PNG)

## Try to move your robot.
Isaac Sim provides a convenient utility to control the articulations. Go to Isaac Utils -> Workflows ->Articulation Inspector. Then play with your robot!

![arti_inspector_loc}](/OmniverseGym/Docs/assets/arti_inspector_location.PNG)
![arti_inspector](/OmniverseGym/Docs/assets/controllers_arti_inspector.PNG)
