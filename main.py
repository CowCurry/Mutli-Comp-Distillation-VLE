import numpy as np
from scipy.optimize import minimize
import pandas as pd

class Component:
    def __init__(self, name, molecular_weight, heat_of_vaporization, vapor_pressure_coeff, liquid_density):
        self.name = name
        self.molecular_weight = molecular_weight
        self.heat_of_vaporization = heat_of_vaporization
        self.vapor_pressure_coeff = vapor_pressure_coeff
        self.liquid_density = liquid_density

class DistillationColumn:
    def __init__(self, components, feed, reflux_ratio, number_of_stages, feed_stage, condenser_type):
        self.components = components
        self.feed = feed
        self.reflux_ratio = reflux_ratio
        self.number_of_stages = number_of_stages
        self.feed_stage = feed_stage
        self.condenser_type = condenser_type
        self.relative_volatility = self.calculate_relative_volatility()
        self.vle_data = self.generate_vle_data()
        self.vlle_data = self.generate_vlle_data()
        
    def calculate_relative_volatility(self):
        volatilities = {}
        for comp in self.components:
            volatilities[comp.name] = np.exp(-comp.vapor_pressure_coeff)
        return volatilities
    
    def generate_vle_data(self):
        data = {}
        for comp in self.components:
            data[comp.name] = []
            for T in range(300, 400, 10):
                P = 101.325 * np.exp(comp.vapor_pressure_coeff * (T - 300)/100)
                data[comp.name].append({'Temperature': T, 'Pressure': P})
        return data
    
    def generate_vlle_data(self):
        data = {}
        for i, comp1 in enumerate(self.components):
            for j, comp2 in enumerate(self.components):
                if i < j:
                    key = f"{comp1.name}-{comp2.name}"
                    data[key] = []
                    for T in range(300, 400, 10):
                        P = 101.325 * np.exp((comp1.vapor_pressure_coeff + comp2.vapor_pressure_coeff)/2 * (T - 300)/100)
                        data[key].append({'Temperature': T, 'Pressure': P})
        return data
    
    def equilibrium_ratios(self, x, T):
        ratios = {}
        for comp in self.components:
            ratios[comp.name] = self.relative_volatility[comp.name] * x[comp.name]
        return ratios
    
    def cost_function(self, reflux_ratio):
        total_cost = 0
        for stage in range(1, self.number_of_stages + 1):
            stage_cost = reflux_ratio * stage
            total_cost += stage_cost
        return total_cost
    
    def optimize_reflux_ratio(self):
        result = minimize(self.cost_function, self.reflux_ratio, bounds=[(1.0, 10.0)])
        self.reflux_ratio = result.x[0]
        return self.reflux_ratio
    
    def simulate(self):
        optimized_reflux = self.optimize_reflux_ratio()
        stages = []
        for stage in range(1, self.number_of_stages + 1):
            stage_data = {'Stage': stage, 'Reflux Ratio': optimized_reflux}
            stages.append(stage_data)
        return pd.DataFrame(stages)

def main():
    components = [
        Component('A', 78.11, 35.69, 0.5, 0.789),
        Component('B', 92.14, 38.25, 0.6, 0.846),
        Component('C', 114.23, 45.10, 0.7, 0.892)
    ]
    feed = {'A': 0.5, 'B': 0.3, 'C': 0.2}
    column = DistillationColumn(components, feed, reflux_ratio=1.5, number_of_stages=20, feed_stage=10, condenser_type='total')
    simulation_results = column.simulate()
    simulation_results.to_csv('distillation_results.csv', index=False)
    
if __name__ == "__main__":
    main()
