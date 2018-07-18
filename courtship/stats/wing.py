# -*- coding: utf-8 -*-

"""
.. module:: statistics
   :synopsis: Functions for determining statistics about
			  wing positioning.

.. moduleauthor:: Ross McKinney
"""

import numpy as np

def _distances(lw, rw, c):
	"""Calculates distances between all three sides of a triangle 
	defined by lw, rw, and c.

	Parameters
	----------
	lw : np.ndarray | shape = [N, 2]
		Left wing centroid positions.

	rw : np.ndarray | shape = [N, 2]
		Right wing centroid positions.

	c : np.ndarray | shape = [N, 2]
		Ellipse centroid positions.

	Returns 
	-------
	a : np.ndarray | shape = [N]
		Distance between rw and c.

	b : np.ndarray | shape = [N]
		Distance between lw and c.

	c : np.ndarray | shape = [N]
		Distance between rw and lw.
	"""
	a = np.sqrt(np.sum((rw - c)**2, axis = 1).astype(np.float))
	b = np.sqrt(np.sum((lw - c)**2, axis = 1).astype(np.float))
	c = np.sqrt(np.sum((lw - rw)**2, axis = 1).astype(np.float))
	return a, b, c

def full_wing_angle(fly):
	"""Calculates the angle defined by left_wing > centroid > right_wing.

	Parameters
	----------
	fly : Fly object

	Returns
	-------
	thetas : np.ndarray | shape = [fly.n_frames]
		Angles (in radians) made between the connected vertices: fly.left_wing >
		fly.centroid > fly.right_wing.
	"""
	centroids = fly.body.centroid.coords_xy()
	l_wing = fly.left_wing.centroid.coords_xy()
	r_wing = fly.right_wing.centroid.coords_xy()

	a, b, c = _distances(l_wing, r_wing, centroids)
	thetas = np.arccos((a**2 + b**2 - c**2)/(2*a*b))
	return thetas

def individual_wing_angles(fly):
	"""Calculates the angles defined by left_wing > centroid > x-axis
	and right_wing > centroid > x-axis.

	Parameters
	----------
	fly : Fly object

	Returns
	-------
	left_thetas : np.ndarray | shape = [fly.n_frames]
		Angle (in radians) between left wing and x-axis.

	right_thetas : np.ndarray | shape = [fly.n_frames]
		Angle (in radians) between right wing and x-axis.
	"""
	centroids = fly.body.centroid.coords_xy()
	#subtract off the centroids (ie make origin), so that we are rotating around the origin.
	l_wing = fly.left_wing.centroid.coords_xy() - centroids
	r_wing = fly.right_wing.centroid.coords_xy() - centroids
	rotations = fly.body.rotation_angle

	rot_l = np.zeros_like(l_wing) #shape = [N, 2]
	rot_r = np.zeros_like(r_wing) #shape = [N, 2]
	for i, rot in enumerate(rotations):
		angle = -rot * np.pi / 180.
		R = np.array([[np.cos(angle), -np.sin(angle)],
					  [np.sin(angle), np.cos(angle)]])
		original_coords = np.vstack((l_wing[i, :], r_wing[i, :]))
		rotated_coords = np.dot(R, original_coords.T).T
		rot_l[i, :] = rotated_coords[0, :]
		rot_r[i, :] = rotated_coords[1, :]

	left_thetas = np.arctan2(rot_l[:, 1], rot_l[:, 0]) + np.pi
	right_thetas = np.abs(np.arctan2(rot_r[:, 1], rot_r[:, 0]) - np.pi)
	return left_thetas, right_thetas

def wing_distances(fly):
	"""Calculates the distances between centroids of left and right wings.

	Parameters
	----------
	fly : Fly object

	Returns
	-------
	distances : np.ndarray | shape = [fly.n_frames]
		Distance (in pixels) between fly.left_wing and fly.right_wing for 
		each frame.
	"""
	centroids = fly.body.centroid.coords_xy()
	l_wing = fly.left_wing.centroid.coords_xy()
	r_wing = fly.right_wing.centroid.coords_xy()

	d = _distances(l_wing, r_wing, centroids)
	return d[2]

if __name__ == "__main__":
	pass


