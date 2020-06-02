import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def showGrid(grid, nbv = None, predicted_nbv = None):
    # receives a plain grid and plots the 3d voxel map
    grid3d = np.reshape(grid, (32,32,32))

    unknown = (grid3d == 0.5)
    occupied = (grid3d > 0.5)

    # combine the objects into a single boolean array
    voxels = unknown | occupied

    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[unknown] = 'yellow'
    colors[occupied] = 'blue'

    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
     
    # plot the NBV
    # the view sphere was placed at 0.4 m from the origin, the voxelmap has an aproximated size of 0.25
    scale = 32/2
    rate_voxel_map_sphere = 0.25
    center = np.ones(3) * scale
    
    if nbv is not None:
        position = nbv[:3]
        position = position * (scale / rate_voxel_map_sphere) + center
        direction = center - position
        #print(position)
        ax.quiver(position[0], position[1], position[2], direction[0], direction[1], direction[2], length=5.0, normalize=True, color = 'g')  
    
    if predicted_nbv is not None:
        position = predicted_nbv[:3]
        position = position * (scale / rate_voxel_map_sphere) + center
        direction = center - position
        #print(position)
        ax.quiver(position[0], position[1], position[2], direction[0], direction[1], direction[2], length=5.0, normalize=True, color = 'r')
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    #plt.show()
    
    
def showGrid4(grid, nbv = None, predicted_nbv = None):
     # receives a plain grid and plots the 3d voxel map
    grid3d = np.reshape(grid, (32,32,32))

    unknown = (grid3d == 0.5)
    occupied = (grid3d > 0.5)

    # combine the objects into a single boolean array
    voxels = unknown | occupied

    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[unknown] = 'yellow'
    colors[occupied] = 'blue'
    
    # plot the NBV
    # the view sphere was placed at 0.4 m from the origin, the voxelmap has an aproximated size of 0.25
    scale = 32/2
    rate_voxel_map_sphere = 0.25
    center = np.ones(3) * scale
    
    if nbv is not None:
        position = nbv[:3]
        position_nbv = position * (scale / rate_voxel_map_sphere) + center
        direction_nbv = center - position_nbv
    if predicted_nbv is not None:
        position = predicted_nbv[:3]
        position_pred = position * (scale / rate_voxel_map_sphere) + center
        direction_pred = center - position_pred    
    
    fig = plt.figure(figsize=(12, 12))
    
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    if nbv is not None:
        ax.quiver(position_nbv[0], position_nbv[1], position_nbv[2], direction_nbv[0], direction_nbv[1], direction_nbv[2], length=5.0, normalize=True, color = 'g') 
    if predicted_nbv is not None:
        ax.quiver(position_pred[0], position_pred[1], position_pred[2], direction_pred[0], direction_pred[1], direction_pred[2], length=5.0, normalize=True, color = 'r')

        
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    if nbv is not None:
        ax.quiver(position_nbv[0], position_nbv[1], position_nbv[2], direction_nbv[0], direction_nbv[1], direction_nbv[2], length=5.0, normalize=True, color = 'g') 
    if predicted_nbv is not None:
        ax.quiver(position_pred[0], position_pred[1], position_pred[2], direction_pred[0], direction_pred[1], direction_pred[2], length=5.0, normalize=True, color = 'r')
    ax.view_init(elev=0.0, azim=0.0)

 
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    if nbv is not None:
        ax.quiver(position_nbv[0], position_nbv[1], position_nbv[2], direction_nbv[0], direction_nbv[1], direction_nbv[2], length=5.0, normalize=True, color = 'g') 
    if predicted_nbv is not None:
        ax.quiver(position_pred[0], position_pred[1], position_pred[2], direction_pred[0], direction_pred[1], direction_pred[2], length=5.0, normalize=True, color = 'r')
    ax.view_init(elev=0.0, azim=90.0)
    
 
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    if nbv is not None:
        ax.quiver(position_nbv[0], position_nbv[1], position_nbv[2], direction_nbv[0], direction_nbv[1], direction_nbv[2], length=5.0, normalize=True, color = 'g') 
    if predicted_nbv is not None:
        ax.quiver(position_pred[0], position_pred[1], position_pred[2], direction_pred[0], direction_pred[1], direction_pred[2], length=5.0, normalize=True, color = 'r')
    ax.view_init(elev=90.0, azim=0.0)

    # # helpful function that displays the octree and the nbv
# def visualize_output(test_grids, test_outputs, gt_nbvs=None, batch_size=batch_size):
#     for i in range(batch_size):

#         # un-transform the image data
#         grid = test_grids[i].data   # get the image from it's wrapper
#         grid = grid.numpy()   # convert to numpy array from a Tensor

#         print(test_outputs[i].data.numpy())
#         # un-transform the predicted nbv
#         predicted = getPosition( test_outputs[i].data.numpy(), nbv_positions)
#         #print(predicted)
        
#         print(gt_nbvs[i].numpy())
#         gt = getPosition(gt_nbvs[i].numpy(), nbv_positions)
#         #print(gt)
        
#         cnbv.showGrid(grid, gt, predicted)
#         #plt.show()
        
#         if i==2:
#             break

# visualize_output(test_grids, test_outputs, np.squeeze(gt_nbvs))