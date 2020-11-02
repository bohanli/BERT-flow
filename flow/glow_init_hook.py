import tensorflow as tf


class GlowInitHook(tf.estimator.SessionRunHook):
  """
  Hook that runs data-dependent initialization once before the first step.

  The init op is stored in the tf collection glow_init_op. Look at the
  "body" in glow.py for more details.
  """

  def after_create_session(self, session, coord):
    del coord
    global_step = session.run(tf.compat.v1.train.get_or_create_global_step())
    if global_step == 0:
      ddi = tf.get_collection("glow_init_op")
      # In-case of a multi-GPU system, this just runs the first op in the
      # collection.
      print(ddi)
      if ddi:
        session.run(ddi[0])
        #input()
