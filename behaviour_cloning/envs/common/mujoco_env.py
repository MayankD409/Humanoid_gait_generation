import contextlib
import os
import numpy as np
import mujoco
import mujoco_viewer

DEFAULT_SIZE = 500

class MujocoEnv():
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, sim_dt, control_dt):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            raise Exception("Provide full path to robot description package.")
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        print(f"Loading MuJoCo model from: {fullpath}")
        try:
            print("Loading model using MjSpec...")
            self.spec = mujoco.MjSpec()
            self.spec.from_file(fullpath)
            self.model = self.spec.compile()
            print(f"Model loaded successfully. Number of bodies: {self.model.nbody}")
            
            print(f"Model has {self.model.nbody} bodies, {self.model.nq} position coordinates, {self.model.nv} velocity coordinates, and {self.model.nu} actuators")
            
            print(f"Detail of bodies:")
            for i in range(self.model.nbody):
                print(f"Body {i}: {self.model.body(i).name}")
                
            print(f"Detail of actuators:")
            for i in range(self.model.nu):
                try:
                    print(f"Actuator {i}: {self.model.actuator(i).name}")
                except Exception as e:
                    print(f"Error accessing actuator {i}: {e}")
                    
            # Try alternative loading method if only world body is found
            if self.model.nbody <= 1:
                print("Only world body found. Trying direct load via mujoco.MjModel.from_xml_path...")
                try:
                    alt_model = mujoco.MjModel.from_xml_path(fullpath)
                    print(f"Alternative loading - Number of bodies: {alt_model.nbody}")
                    for i in range(alt_model.nbody):
                        print(f"Alt Body {i}: {alt_model.body(i).name}")
                    if alt_model.nbody > 1:
                        print("Using alternatively loaded model instead")
                        self.model = alt_model
                except Exception as alt_e:
                    print(f"Alternative loading also failed: {alt_e}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # set frame skip and sim dt
        self.frame_skip = (control_dt/sim_dt)
        self.model.opt.timestep = sim_dt

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.lookat[2] = 1.5
        self.viewer.cam.lookat[0] = 2.0
        self.viewer.cam.elevation = -20
        self.viewer.vopt.geomgroup[2] = 0
        self.viewer._render_every_frame = True

    def viewer_is_paused(self):
        return self.viewer._paused

    # -----------------------------
    # (some methods are taken directly from dm_control)

    @contextlib.contextmanager
    def disable(self, *flags):
      """Context manager for temporarily disabling MuJoCo flags.

      Args:
        *flags: Positional arguments specifying flags to disable. Can be either
          lowercase strings (e.g. 'gravity', 'contact') or `mjtDisableBit` enum
          values.

      Yields:
        None

      Raises:
        ValueError: If any item in `flags` is neither a valid name nor a value
          from `mujoco.mjtDisableBit`.
      """
      old_bitmask = self.model.opt.disableflags
      new_bitmask = old_bitmask
      for flag in flags:
        if isinstance(flag, str):
          try:
            field_name = "mjDSBL_" + flag.upper()
            flag = getattr(mujoco.mjtDisableBit, field_name)
          except AttributeError:
            valid_names = [
                field_name.split("_")[1].lower()
                for field_name in list(mujoco.mjtDisableBit.__members__)[:-1]
            ]
            raise ValueError("'{}' is not a valid flag name. Valid names: {}"
                             .format(flag, ", ".join(valid_names))) from None
        elif isinstance(flag, int):
          flag = mujoco.mjtDisableBit(flag)
        new_bitmask |= flag.value
      self.model.opt.disableflags = new_bitmask
      try:
        yield
      finally:
        self.model.opt.disableflags = old_bitmask

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,), \
            f"qpos shape {qpos.shape} is expected to be {(self.model.nq,)}"
        assert qvel.shape == (self.model.nv,), \
            f"qvel shape {qvel.shape} is expected to be {(self.model.nv,)}"
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.act = []
        self.data.plugin_state = []
        # Disable actuation since we don't yet have meaningful control inputs.
        with self.disable('actuation'):
            mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer_setup()
        self.viewer.render()

    def uploadGPU(self, hfieldid=None, meshid=None, texid=None):
        # hfield
        if hfieldid is not None:
            mujoco.mjr_uploadHField(self.model, self.viewer.ctx, hfieldid)
        # mesh
        if meshid is not None:
            mujoco.mjr_uploadMesh(self.model, self.viewer.ctx, meshid)
        # texture
        if texid is not None:
            mujoco.mjr_uploadTexture(self.model, self.viewer.ctx, texid)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
