# How to create rigid body objects for RL training

The guide aims to explain the pipeline of creating rigid body objects with no DoF of actuation or motion (i.e. goal indicator, objects being manipulated) in omniverse.

## Pre-request

You should has the mesh file of the rigid object. Suggested file format is obj.

## Importing pipeline

1. Open Isaac Sim;
1. Create an empty stage by clicking File -> New From Stage Template -> Empty;
1. Delete the prim "World (defaultPrim)" in the stage tree at the right hand of the UI;
1. Click File -> Import;
1. Select the mesh file in the pop-up window;
1. After the last step, the current displayed USD file is a reference to the original USD file. The original USD file is on the same directory of the mesh file. Open this original USD file and leave the current file unsaved.
1. Change the name of the prim "World (defaultPrim)" to the same name of the mesh file (child of World (defaultPrim))
1. In the stage tree, click on the root prim (the prim whose name was changed in the last step). Click the green "Add" button, and click Physics -> Articulation Root;
1. Save the USD and the USD file can be used as a rigid object in your reinforcement learning environment.

## Some Tips

### Floating objects

Sometimes we want the objects not to move but stay on the same location within the episode. To do this, open the USD file, right click the child prim, select create -> Physics -> Joint -> Fixed Joint.

### Objects with no collision

If the objects is not expected to have physical contact with other objects, for example when the object is simply served as the indicator of the goal, then you can deactivate the collider of the meshes:

1. Span the child prim, and you can see all the child meshes of the current USD file.
1. Select all the meshes, in the "Property" window, at "Physics" -> "Collider", de-select the checkbox "Collision Enabled".