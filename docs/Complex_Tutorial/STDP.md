#STDP

This module allows synapses to adjust their strengths based on the temporal relation between the pre- and post-synaptic voltages.
The core of the learning rule consists of a discrete kernel with three elements.
Each element compares pre- and postsynaptic voltages across subsequent, pre_post or post_pre, or simultaneous time steps, simu (see figure).
The global learning rate is defined by stdp_factor
After modifying synapses for which plasticity has been enabled, s.enabled, synaptic weights are clipped to values between 0 and 10.
Since the weight change depends on the temporal relation between current and preceding activities, we store activities of the preceding time step in the \textit{voltage\_old}.

This stdp module is a NeuronGroup module, therefore it iterates over all the synapses of a given NeuronGroup of a specific type (GLUTAMATE).
By removing the for-loop, a similar module could, however, also be added to the behaviour of a SynapseGroups directly. 

<img width="300" src="https://raw.githubusercontent.com/trieschlab/PymoNNto/Images/STDP_beh.png"><img width="300" src="https://raw.githubusercontent.com/trieschlab/PymoNNto/Images/STDP_vg.png"><br>

```python

from PymoNNto.NetworkCore.Behaviour import *

class STDP(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('STDP')
        neurons.stdp_factor = self.get_init_attr('stdp_factor', 0.00015, neurons)
        self.syn_type = self.get_init_attr('syn_type', 'GLUTAMATE', neurons)
        neurons.voltage_old = neurons.get_neuron_vec()

    def new_iteration(self, neurons):

        for s in neurons.afferent_synapses[self.syn_type]:

            pre_post = s.dst.voltage[:, None] * s.src.voltage_old[None, :]
            simu = s.dst.voltage[:, None] * s.src.voltage[None, :]
            post_pre = s.dst.voltage_old[:, None] * s.src.voltage[None, :]

            dw = neurons.stdp_factor * (pre_post - post_pre + simu)

            s.W = np.clip(s.W+dw*s.enabled, 0.0, 10.0)

        neurons.voltage_old = neurons.voltage.copy()


```