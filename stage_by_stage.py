import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Component:
    def __init__(self, name, molecular_weight, heat_of_vaporization, vapor_pressure_coeff, liquid_density, specific_heat_liquid, specific_heat_vapor):
        self.name = name
        self.molecular_weight = molecular_weight
        self.heat_of_vaporization = heat_of_vaporization
        self.vapor_pressure_coeff = vapor_pressure_coeff
        self.liquid_density = liquid_density
        self.specific_heat_liquid = specific_heat_liquid
        self.specific_heat_vapor = specific_heat_vapor

class VLEData:
    def __init__(self, components, temperatures, pressures, mole_fractions):
        self.components = components
        self.temperatures = temperatures
        self.pressures = pressures
        self.mole_fractions = mole_fractions

    def get_pressure(self, T):
        P = {}
        for comp in self.components:
            P[comp.name] = 101.325 * np.exp(comp.vapor_pressure_coeff * (T - 300)/100)
        return P

class DistillationStage:
    def __init__(self, stage_number, feed, reflux, distillate, bottoms, temperature, pressure, compositions):
        self.stage_number = stage_number
        self.feed = feed
        self.reflux = reflux
        self.distillate = distillate
        self.bottoms = bottoms
        self.temperature = temperature
        self.pressure = pressure
        self.compositions = compositions

class DistillationColumn:
    def __init__(self, components, feed, num_stages, feed_stage, condenser_type, reboiler_type):
        self.components = components
        self.feed = feed
        self.num_stages = num_stages
        self.feed_stage = feed_stage
        self.condenser_type = condenser_type
        self.reboiler_type = reboiler_type
        self.vle = self.generate_vle_data()
        self.stages = []
        self.cost = 0

    def generate_vle_data(self):
        temperatures = np.linspace(300, 400, 11)
        pressures = []
        mole_fractions = []
        for T in temperatures:
            P = {}
            x = {}
            for comp in self.components:
                P[comp.name] = 101.325 * np.exp(comp.vapor_pressure_coeff * (T - 300)/100)
                x[comp.name] = np.random.uniform(0.1, 0.9)
            pressures.append(P)
            mole_fractions.append(x)
        return VLEData(self.components, temperatures, pressures, mole_fractions)

    def equilibrium_ratio(self, T):
        ratios = {}
        for comp in self.components:
            ratios[comp.name] = self.vle.get_pressure(T)[comp.name] / 101.325
        return ratios

    def energy_balance(self, stage):
        Q = 0
        for comp in self.components:
            Q += stage.compositions[comp.name]['vapor'] * comp.specific_heat_vapor * stage.temperature
            Q -= stage.compositions[comp.name]['liquid'] * comp.specific_heat_liquid * stage.temperature
        return Q

    def mass_balance(self, stage):
        total_vapor = sum([comp['vapor'] for comp in stage.compositions.values()])
        total_liquid = sum([comp['liquid'] for comp in stage.compositions.values()])
        return total_vapor, total_liquid

    def cost_function(self, reflux_ratio):
        total_cost = 0
        for stage in range(1, self.num_stages + 1):
            stage_cost = reflux_ratio * stage
            total_cost += stage_cost
        return total_cost

    def optimize_reflux_ratio(self):
        result = minimize(self.cost_function, 1.5, bounds=[(1.0, 10.0)])
        return result.x[0]

    def simulate_stage(self, stage_number, reflux_ratio):
        T = 300 + stage_number * 10
        P = 101.325
        ratios = self.equilibrium_ratio(T)
        compositions = {}
        for comp in self.components:
            vapor = ratios[comp.name] * self.feed[comp.name] * reflux_ratio
            liquid = self.feed[comp.name] - vapor
            compositions[comp.name] = {'vapor': vapor, 'liquid': liquid}
        stage = DistillationStage(stage_number, self.feed, reflux_ratio, 0, 0, T, P, compositions)
        Q = self.energy_balance(stage)
        total_vapor, total_liquid = self.mass_balance(stage)
        self.cost += Q
        self.stages.append(stage)

    def simulate(self):
        optimized_reflux = self.optimize_reflux_ratio()
        for stage in range(1, self.num_stages + 1):
            self.simulate_stage(stage, optimized_reflux)
        data = []
        for stage in self.stages:
            data.append({
                'Stage': stage.stage_number,
                'Temperature (K)': stage.temperature,
                'Pressure (kPa)': stage.pressure,
                'Reflux Ratio': optimized_reflux,
                'Energy Balance': self.energy_balance(stage),
                'Total Vapor Flow': self.mass_balance(stage)[0],
                'Total Liquid Flow': self.mass_balance(stage)[1]
            })
        df = pd.DataFrame(data)
        df.to_csv('stage_by_stage_results.csv', index=False)
        self.plot_results(df)

    def plot_results(self, df):
        plt.figure(figsize=(10,6))
        plt.plot(df['Stage'], df['Temperature (K)'], label='Temperature')
        plt.plot(df['Stage'], df['Pressure (kPa)'], label='Pressure')
        plt.xlabel('Stage')
        plt.ylabel('Values')
        plt.title('Stage-by-Stage Temperature and Pressure')
        plt.legend()
        plt.savefig('stage_by_stage_plot.png')
        plt.close()

def main():
    components = [
        Component('A', 78.11, 35.69, 0.5, 0.789, 1.2, 2.0),
        Component('B', 92.14, 38.25, 0.6, 0.846, 1.3, 2.1),
        Component('C', 114.23, 45.10, 0.7, 0.892, 1.4, 2.2)
    ]
    feed = {'A': 0.5, 'B': 0.3, 'C': 0.2}
    column = DistillationColumn(components, feed, num_stages=30, feed_stage=15, condenser_type='partial', reboiler_type='steam')
    column.simulate()

if __name__ == "__main__":
    main()
