namespace RMV.ML.Network.Domain;

#region LayerType -----------------------------------------------------

/// <summary>
/// Type of the layer
/// </summary>
public enum LayerType
{
	Unknown,
	Input,
	Hidden,
	Output
}

#endregion


#region InitType ------------------------------------------------------

/// <summary>
/// Neuron initialization type
/// </summary>
public enum InitType
{
	Unknown,
	N_W,
	Random,
	RndZero
}

#endregion


#region Learning Type -------------------------------------------------

public enum LearningType
{
   Unknown,
	Batch,
	Online,     
}

#endregion
