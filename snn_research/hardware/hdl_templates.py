# ファイルパス: snn_research/hardware/hdl_templates.py
# Title: Verilog HDL Templates for SNN Hardware
# Description:
# - LIFニューロンとSTDP学習則のハードウェア記述テンプレート。
# - Phase 6の「FPGA/ASIC対応」を実現するための基盤。

LIF_NEURON_VERILOG = """
module lif_neuron #(
    parameter THRESHOLD = 8'd128,
    parameter DECAY = 8'd10,
    parameter REST_POTENTIAL = 8'd0
)(
    input wire clk,
    input wire rst_n,
    input wire spike_in,
    input wire [7:0] weight,
    output reg spike_out,
    output reg [7:0] membrane_potential
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            membrane_potential <= REST_POTENTIAL;
            spike_out <= 1'b0;
        end else begin
            // Membrane Potential Update
            // V(t) = V(t-1) - Decay + Input
            if (membrane_potential > DECAY)
                membrane_potential <= membrane_potential - DECAY;
            else
                membrane_potential <= REST_POTENTIAL;

            if (spike_in) begin
                if (membrane_potential + weight > 255) // Saturation check
                    membrane_potential <= 8'd255;
                else
                    membrane_potential <= membrane_potential + weight;
            end

            // Threshold Check & Spike Generation
            if (membrane_potential >= THRESHOLD) begin
                spike_out <= 1'b1;
                membrane_potential <= REST_POTENTIAL; // Reset
            end else begin
                spike_out <= 1'b0;
            end
        end
    end
endmodule
"""

STDP_LEARNING_VERILOG = """
module stdp_synapse #(
    parameter LEARNING_RATE = 8'd2,
    parameter TIME_WINDOW = 8'd20
)(
    input wire clk,
    input wire rst_n,
    input wire pre_spike,
    input wire post_spike,
    output reg [7:0] weight
);
    reg [7:0] last_pre_time;
    reg [7:0] last_post_time;
    reg [7:0] current_time;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight <= 8'd128; // Initial weight
            last_pre_time <= 8'd0;
            last_post_time <= 8'd0;
            current_time <= 8'd0;
        end else begin
            current_time <= current_time + 1;

            if (pre_spike) begin
                last_pre_time <= current_time;
                // Depression (Post -> Pre)
                if ((current_time - last_post_time) < TIME_WINDOW) begin
                    if (weight > LEARNING_RATE)
                        weight <= weight - LEARNING_RATE;
                end
            end

            if (post_spike) begin
                last_post_time <= current_time;
                // Potentiation (Pre -> Post)
                if ((current_time - last_pre_time) < TIME_WINDOW) begin
                    if (weight < (255 - LEARNING_RATE))
                        weight <= weight + LEARNING_RATE;
                end
            end
        end
    end
endmodule
"""

CUDA_KERNEL_TEMPLATE = """
extern "C" __global__
void event_driven_lif_kernel(
    const float* __restrict__ weights,
    const int* __restrict__ spike_indices,
    float* __restrict__ membrane_potentials,
    int* __restrict__ output_spikes,
    int num_neurons,
    int num_spikes,
    float threshold,
    float decay
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    // Decay
    float v = membrane_potentials[idx];
    v *= decay;

    // Integrate Input Spikes (Sparse)
    // Simplified for demonstration: Assuming pre-synaptic mapping is handled
    for (int i = 0; i < num_spikes; ++i) {
        int input_idx = spike_indices[i];
        // Fetch weight for (input_idx -> idx)
        // Note: Real implementation needs CSR/CSC format for weights
        float w = weights[idx * num_neurons + input_idx]; 
        v += w;
    }

    // Fire & Reset
    if (v >= threshold) {
        output_spikes[idx] = 1;
        v = 0.0f;
    } else {
        output_spikes[idx] = 0;
    }
    
    membrane_potentials[idx] = v;
}
"""