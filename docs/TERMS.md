# Mosaic Terminology

These terms are provisional but give the repository a common language.

**Mosaic**  
The complete logical agent architecture and its total capability pool.

**Boot / Core**  
The persistent coordination component descended from the original `VERA-MODEL20B_boot` idea. "Boot" preserves the originating terminology; "Core" emphasizes that its role may extend beyond startup.

**Specialist**  
A separately loadable learned capability module optimized for a bounded class of work.

**Partition**  
A physical or logical division of the larger learned system. A partition is not assumed to be a clean slice of an existing dense transformer.

**Capability pool**  
All learned parameters and callable specialist capabilities available to Mosaic, whether or not they are presently loaded.

**Resident set**  
The parameters and runtime state currently occupying accelerator memory.

**Whole-agent mode**  
Operation in which multiple Mosaic capabilities compose through shared state as one logical agent. This does not necessarily mean all parameters are simultaneously resident.

**Swap**  
Eviction of one specialist from accelerator memory and loading of another.

**Handoff**  
Transfer of task-relevant state from one specialist stage to another.

**Shared state**  
The continuity-bearing information available across specialist boundaries. It may include durable memory, structured task state, artifacts, or learned latent state.

**Currentness**  
The property that the system uses the presently valid fact/decision/state rather than a superseded one.

**Drift**  
Loss of alignment with established current state over time or across handoffs, including reversion to superseded information.

**User correction burden**  
The number of times a user must re-supply, correct, or re-establish information that the system should already have preserved.

**Teacher**  
A larger model used to train, distill, supervise, or align Mosaic components. A teacher is a training role, not necessarily a runtime dependency.
