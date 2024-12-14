import starsim as ss
import typhoidsim as ty
import matplotlib.pyplot as plt

# Define the parameters
pars = dict(
    n_agents = 10e3,
    start = 2000,
    n_years = 2,
    dt = 1.0/ty.days_per_year,
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