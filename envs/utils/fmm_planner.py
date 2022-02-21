import cv2
import numpy as np
import skfmm
import skimage
from numpy import ma
import pickle
import os


def get_mask(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size))

    mask[0, size//2] = 1
    mask[-1, size//2] = 1
    mask[size//2, 0] = 1
    mask[size//2, -1] = 1

    mask[0, 0] = 1
    mask[-1, -1] = 1
    mask[0, -1] = 1
    mask[-1, 0] = 1

    mask[size//2, size//2] = 1
    return mask


def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for i in range(size):
        for j in range(size):
                mask[i, j] = max(5, (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                 ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    return mask


class FMMPlanner():
    def __init__(self, traversible, args, scale=1, step_size=5):
        self.scale = scale
        self.args = args
        self.step_size = step_size
        self.visualize = args.visualize
        self.save = args.save_pictures
        self.save_t = 0
        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale,
                                           traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.))

    def set_goal(self, goal, auto_improve=False):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0] / (self.scale * 1.)), \
                         int(goal[1] / (self.scale * 1.))

        if self.traversible[goal_x, goal_y] == 0. and auto_improve:
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y])

        traversible_ma[goal_x, goal_y] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        return

    def set_multi_goal(self, goal_map):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        traversible_ma[goal_map == 1] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        if self.visualize:
            cv2.imshow("fmmdist", self.fmm_dist / self.fmm_dist.max())
            cv2.waitKey(1)
            
        if self.save :
            if not os.path.exists("fmm_dist_weird/"):
                os.makedirs("fmm_dist_weird/")
            cv2.imwrite("fmm_dist_weird/"+ "fmm_dist_" + str(self.save_t) + ".png", self.fmm_dist) 
            self.save_t +=1 
        return

    def get_short_term_goal(self, state, found_goal = 0, decrease_stop_cond=0):
        scale = self.scale * 1.
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask = get_mask(dx, dy, scale, self.step_size)
        dist_mask = get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]

        dist = np.pad(self.fmm_dist, self.du,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        subset = dist[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]

        assert subset.shape[0] == 2 * self.du + 1 and \
               subset.shape[1] == 2 * self.du + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2
        
        stop_condition = max((self.args.stop_cond - decrease_stop_cond)*100/5., 0.2)
        if not(found_goal): 
            stop_condition = stop_condition
        else:
            stop_condition = stop_condition
            
        if self.visualize:
            print("dist until goal is ", subset[self.du, self.du])
        if subset[self.du, self.du] <stop_condition:
            stop = True
        else:
            stop = False

        subset -= subset[self.du, self.du]
        ratio1 = subset / dist_mask
        subset[ratio1 < -1.5] = 1

        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        if subset[stg_x, stg_y] > -0.0001:
            replan = True
        else:
            replan = False


        return (stg_x + state[0] - self.du) * scale, \
               (stg_y + state[1] - self.du) * scale, replan, stop

    def _find_nearest_goal(self, goal):
        traversible = skimage.morphology.binary_dilation(
                                    np.zeros(self.traversible.shape),
                                    skimage.morphology.disk(2)) != True
        traversible = traversible*1.
        planner = FMMPlanner(traversible)
        planner.set_goal(goal)

        mask = self.traversible

        dist_map = planner.fmm_dist * mask
        dist_map[dist_map == 0] = dist_map.max()

        goal = np.unravel_index(dist_map.argmin(), dist_map.shape)

        return goal
