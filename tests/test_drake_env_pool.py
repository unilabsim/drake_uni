from __future__ import annotations

import pytest


def test_one_dof_layout_binds_global_names_when_importer_order_differs(tmp_path):
    from drake_uni.runtime import DrakeBatchConfig, batch_diagnostics, create_runtime

    if not batch_diagnostics().batch_available:
        pytest.skip("Drake native batch extension is not available")

    scene = tmp_path / "scene.xml"
    scene.write_text(
        '<mujoco><worldbody>'
        '<body name="object/base"><freejoint name="object/root"/>'
        '<body name="object/link"><joint name="object/hinge"/>'
        '<geom name="object/link_geom" size=".05"/></body></body>'
        '<body name="robot/base"><body name="robot/link">'
        '<joint name="robot/hinge"/><geom name="robot/link_geom" size=".05"/>'
        "</body></body></worldbody>"
        '<actuator><motor name="drive" joint="robot/hinge"/></actuator></mujoco>',
        encoding="utf-8",
    )
    runtime = create_runtime(DrakeBatchConfig(str(scene), 1, 0.001))
    try:
        assert runtime.model_info().joint_names == (
            "object/root",
            "object/hinge",
            "robot/hinge",
        )
    finally:
        runtime.close()
