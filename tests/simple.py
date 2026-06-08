import starsim as ss
import typhoidsim as ty
import matplotlib.pyplot as plt

# Define the parameters
pars = dict(
    n_agents = 10e3,
    start = 2000,
    dur = 2,
    dt = 1/365,
)

typh = ty.Typhoid(dt=ss.years(1/365))

sim = ss.Sim(
    pars = pars,
    diseases = typh,
    networks = 'random',
)

sim.run()
sim.plot()
sim.summarize()
plt.show()