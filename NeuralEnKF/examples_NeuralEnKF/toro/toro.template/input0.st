//Units: non-dimensional. This is just the Toro tube problem

under Mesh {
  Type = ThreeDimensional;
  X0   = 0.0;
  Xmax = 1.0;
  Y0   = -0.05;
  Ymax = 0.05;
  Z0   = -0.05;
  Zmax = 0.05;
  NumberOfCellsX = 400;
  NumberOfCellsY = 1;
  NumberOfCellsZ = 1;

  BoundaryConditionX0   = Inlet;
  BoundaryConditionXmax = Inlet2;
  BoundaryConditionY0   = Symmetry;
  BoundaryConditionYmax = Symmetry;
  BoundaryConditionZ0   = Symmetry;
  BoundaryConditionZmax = Symmetry;
}

under Equations {
  under Material[0] {
    EquationOfState = NobleAbelStiffenedGas;
    under NobleAbelStiffenedGasModel {
      SpecificHeatRatio = 1.4;
      PressureConstant = 0.0;
      VolumeConstant = 0.0;
      EnergyConstant = 0.0;

      SpecificHeatAtConstantVolume = 2.5;
    }
  }
  under Material[1] {
    EquationOfState = StiffenedGas;
    under StiffenedGasModel {
      SpecificHeatRatio = 1.4;
      PressureConstant = 0.0;
    }
  }
}

under InitialCondition {

  under GeometricEntities {

    under Plane[0] {
      Point_x = 0.4;
      Point_y = 0.0;
      Point_z = 0.0;
      Normal_x = 1.0;
      Normal_y = 0.0;
      Normal_z = 0.0;

      under InitialState {
        MaterialID = 0;
        Density = 5.99242; 
        VelocityX = -6.19633;
        VelocityY = 0.0;
        VelocityZ = 0.0;
        Pressure = 46.0950;
      }
    }
  }
}

under BoundaryConditions {
  under Inlet {
    MaterialID = 0;
    Density = 5.99924; 
    VelocityX = 19.5975;
    VelocityY = 0.0;
    VelocityZ = 0.0;
    Pressure = 460.894;
  }
  under Inlet2 {
    MaterialID = 0;
    Density = 5.99242; 
    VelocityX = -6.19633;
    VelocityY = 0.0;
    VelocityZ = 0.0;
    Pressure = 46.0950;
  }
}

under Space {
  under NavierStokes {
    Flux = LocalLaxFriedrichs;
    under Reconstruction {
      Type = Linear;    
      VariableType = Primitive; //not very sensitive
      SlopeNearInterface = NonZero;
      //Limiter = GeneralizedMinMod;
      Limiter = VanAlbada;
      GeneralizedMinModCoefficient = 1.2;
    }
  }

  under LevelSet[0] {
    MaterialID = 1;
    under Reconstruction {
      Type = Linear;
      Limiter = None;
    }
  }

}

under MultiPhase {
  Flux = Numerical;
  ReconstructionAtInterface = Linear;  //slightly better than constant
  PhaseChange = RiemannSolution; //clearly better than extrapolation
}

under Time {
  Type = Explicit;
  //MaxIts = 500;
  MaxTime = 0.005;
  //CFL    = 0.8;
  TimeStep = 0.00002;
  under Explicit {
    Type = RungeKutta2;
  }
}

under Output {
  Prefix = "results/";
  Solution = "solution_0000";
  Frequency = 5;
  //TimeInterval = 0.05;
  Density = On;
  Velocity = On;
  Pressure = On;
  LevelSet0 = On;
  MaterialID = On;
  Temperature = On;

  //VerboseScreenOutput = On;
}
