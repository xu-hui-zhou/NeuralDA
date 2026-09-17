
under Mesh {
  Type = ThreeDimensional;
  X0   = 0.0;
  Xmax = 2.0;
  Y0   = 0.0;
  Ymax = 2.0;
  Z0   = -0.001;
  Zmax = 0.001;

  BoundaryConditionX0   = Farfield;
  BoundaryConditionXmax = Farfield;
  BoundaryConditionY0   = Farfield;
  BoundaryConditionYmax = Farfield;
  BoundaryConditionZ0   = Symmetry;
  BoundaryConditionZmax = Symmetry;

  NumberOfCellsX = 400;
  NumberOfCellsY = 400;
  NumberOfCellsZ = 1;
}

under Equations {
  under Material[1] { // liquid (Muller et al. Computer & Fluid, 2009)
    EquationOfState = StiffenedGas;
    under StiffenedGasModel {
      SpecificHeatRatio = 1.1;
      PressureConstant = 2.045e9;
    }
  }
  under Material[0] { // gas bubble  (air)
    EquationOfState = StiffenedGas;
    under StiffenedGasModel {
      SpecificHeatRatio = 1.4;
      PressureConstant = 0.0;
    } 
  }
}

under InitialCondition {
  under GeometricEntities {
    under Sphere[0] {
      Center_x = 1.0;
      Center_y = 1.0;
      Center_z = 0.0;
      Radius = 0.4;
      under InitialState {
        MaterialID = 0;
        Density = 1.0; 
        VelocityX = 0.0;
        VelocityY = 0.0;
        VelocityZ = 0.0;
        Pressure = 1000.0;
      }
    }
  }
}

under BoundaryConditions {
  under Farfield {
    MaterialID = 0;
    Density = 1.0; 
    VelocityX = 0.0;
    VelocityY = 0.0;
    VelocityZ = 0.0;
    Pressure = 0.01;
  }
}

under Space {
  under NavierStokes {
    Flux = LocalLaxFriedrichs; //HLLC; //Roe;
    under Reconstruction {
      Type = Linear;    
      VariableType = Primitive; //Conservative; //ConservativeCharacteristic;
      Limiter = GeneralizedMinMod;
      GeneralizedMinModCoefficient = 1.0; //default: 1.2 (larger --> less dissipation)
    }
  }

  under LevelSet[0] {
    MaterialID = 1;
    Solver = FiniteDifference;

    BoundaryConditionX0   = LinearExtrapolation;
    BoundaryConditionXmax = LinearExtrapolation;
    BoundaryConditionY0   = ZeroNeumann;
    BoundaryConditionYmax = ZeroNeumann;
    BoundaryConditionZ0   = ZeroNeumann;
    BoundaryConditionZmax = ZeroNeumann;
  }
}

under MultiPhase {
  Flux = Numerical;
  ReconstructionAtInterface = Constant;
  PhaseChange = RiemannSolution;
  RiemannNormal = Mesh;
}

under Time {
  Type = Explicit;
  MaxTime = 1.0e-2;
  TimeStep = 4.0e-5;
  //CFL = 0.4;
  under Explicit {
    Type = RungeKutta2;
  }
}


under Output {
  Prefix = "results/";
  Solution = "solution";
  //TimeInterval = 1.0e-4;
  Frequency = 2;
  Velocity = On;
  LevelSet0 = On;
  MaterialID = On;
  Pressure = On;
  Density = On;

  VerboseScreenOutput = Medium;

  under MaterialVolumes { //NOTE: does not account for spherical symmetry!
    FileName = "volumes";
    TimeInterval = 1.0e-7;
  }
}

