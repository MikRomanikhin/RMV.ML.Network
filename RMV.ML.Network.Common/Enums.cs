using System.Text.Json.Serialization;

namespace RMV.ML.Network.Common;

/// <summary>
/// Network learning type
/// </summary>
[JsonConverter( typeof( JsonStringEnumConverter ) )]
public enum LearningType
{
	Unknown,
	Batch,
	MiniBatch,
	Online,
}

[JsonConverter( typeof( JsonStringEnumConverter ) )]
public enum  ActivationType
{
	Relu, Sigmoid
}

[JsonConverter( typeof( JsonStringEnumConverter ) )]
public enum Optimizer
{
	SGD, Momentum, Nesterov, Adam, AdaGrad, RmsProp
}

/// <summary>
/// Neuron initialization type
/// </summary>
//public enum InitType
//{
//	Unknown,	N_W,	N_E,	Random,	RndZero
//}
