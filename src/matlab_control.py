import numpy as np

def pos_controller(ref_pos, pos, pos_pid):
    ref_vel = pos_pid.step(ref_pos, pos, dt=0.01)
    return ref_vel

def vel_leash(ref_vel, eul, max_vel):
    The = eul[1]
    Phi = eul[0]
    Psi = eul[2]
    R_b_e = np.array([[np.cos(Psi)*np.cos(The), np.cos(Psi)*np.sin(The)*np.sin(Phi)-np.sin(Psi)*np.cos(Phi), np.cos(Psi)*np.sin(The)*np.cos(Phi)+np.sin(Psi)*np.sin(Phi)],
                  [np.sin(Psi)*np.cos(The), np.sin(Psi)*np.sin(The)*np.sin(Phi)+np.cos(Psi)*np.cos(Phi), np.sin(Psi)*np.sin(The)*np.cos(Phi)-np.cos(Psi)*np.sin(Phi)],
                  [-np.sin(The), np.cos(The)*np.sin(Phi), np.cos(The)*np.cos(Phi)]])
    
    velbody = R_b_e.T @ ref_vel
    velbody = velbody[0:2]

    norm = np.linalg.norm(velbody)
    if norm > max_vel:
        velbody = velbody * (max_vel / norm)

    velinert = np.linalg.pinv(R_b_e.T) @ np.array([velbody[0], velbody[1], 0])

    return velinert

def vel_controller(ref_vel, vel, vel_pid):
    angle_ref = vel_pid.step(ref_vel, vel, dt=0.01)
    angle_ref[0:2] = np.clip(angle_ref[0:2], -22.5*np.pi/180, 22.5*np.pi/180) # TODO: degrees and radians mixed
    angle_ref[1] *= -1
    angle_ref[2] += 9.8*10.66
    return angle_ref

def angle_controller(theta_phi_ref, eul, angle_p=5):
    angle_ref = np.array([theta_phi_ref[0], theta_phi_ref[1], 0])
    err = angle_ref - eul
    return err * angle_p

def rate_controller(rate_ref, rate, rate_p=8):
    err = rate_ref - rate
    return err * rate_p