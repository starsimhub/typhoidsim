import starsim as ss
import typhoidsim as ty
import matplotlib.pyplot as plt

# Define the parameters
pars = dict(
    n_agents = 10e3,
    start = '2000-01-01',
    stop = '2002-01-01',
    unit = 'day',
    dt = 1.0,
)

typh = ty.Typhoid()

sim = ss.Sim(
    pars = pars,
    diseases = typh,
    networks = 'random',
)

sim.run()
sim.plot()
sim.summarize()
plt.show()