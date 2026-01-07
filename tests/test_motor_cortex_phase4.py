import torch
from snn_research.cognitive_architecture.motor_cortex import MotorCortex


def test_motor_cortex_execution():
    motor = MotorCortex(actuators=['arm_l', 'arm_r'])
    commands = [{'timestamp': 0.1, 'command': 'MOVE_UP'}]
    log = motor.execute_commands(commands)

    assert len(log) == 1
    assert "MOVE_UP" in log[0]


def test_motor_cortex_reflex():
    motor = MotorCortex(device='cpu')
    motor.reflex_enabled = True

    # Random sensory input
    sensory = torch.randn(1, 128)

    # Reflex module initialized randomly, so we can't guarantee a trigger
    # But we can verify it runs without error and returns Int or None
    result = motor.generate_spiking_signal(sensory)
    assert result is None or isinstance(result, int)
