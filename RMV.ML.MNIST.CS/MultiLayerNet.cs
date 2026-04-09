using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

using RMV.ML.Network.Common;

namespace RMV.ML.MNIST.CS;

/// <summary>
/// Fully connected deep neural network with arbitrary hidden layers.
/// </summary>
public class MultiLayerNet
{
	public int InputSize { get; }
	public int OutputSize { get; }
	public List<int> HiddenSizeList { get; }
	public int HiddenLayerNum { get; }
	public double WeightDecayLambda { get; }

	Dictionary<string, object> Params { get; } = [];

	readonly List<(string Name, object Layer)> Layers = [];
	public SoftmaxWithLoss LastLayer { get; } = new();

	readonly IOptimizer optimizer;

	/// <summary>
	/// Construct a multi-layer fully connected neural network based on the provided settings.
	/// </summary>	
	public MultiLayerNet( AppSettings settings )
	{
		this.InputSize = settings.Input;
		this.HiddenSizeList = [ .. settings.Hidden ];
		this.OutputSize = settings.Output;
		this.HiddenLayerNum = settings.Hidden.Length;
		this.WeightDecayLambda = settings.Decay;

		this.optimizer = BuildOptimizer( settings );

		InitWeight( settings.Activation );

		for( int i = 1; i <= HiddenLayerNum; i++ )
		{
			var w = ( Matrix<double> )Params[ "W" + i ];
			var b = ( Vector<double> )Params[ "b" + i ];
			this.Layers.Add( ("Affine" + i, new Affine( w, b )) );

			if( settings.Activation == ActivationType.Relu )
				this.Layers.Add( ("Activation_function" + i, new Relu()) );
			else if( settings.Activation == ActivationType.Sigmoid )
				this.Layers.Add( ("Activation_function" + i, new Sigmoid()) );
		}

		int lastIdx = this.HiddenLayerNum + 1;
		var wLast = ( Matrix<double> )Params[ "W" + lastIdx ];
		var bLast = ( Vector<double> )Params[ "b" + lastIdx ];
		this.Layers.Add( ("Affine" + lastIdx, new Affine( wLast, bLast )) );
	}

	/// <summary>
	/// Creates Optimizer based on configuration settings 
	/// </summary>	
	static IOptimizer BuildOptimizer( AppSettings settings )
	{
		return settings.Optimizer switch {
			Optimizer.SGD => new SGD(),
			Optimizer.Momentum => new Momentum(),
			Optimizer.Adam => new Adam(),
			Optimizer.Nesterov => new Nesterov(),
			Optimizer.AdaGrad => new AdaGrad(),
			Optimizer.RmsProp => new RmsProp(),
			_ => throw new Exception( $"Unsupported optimizer: {settings.Optimizer}" )
		};
	}

	/// <summary>
	/// Initialize weights based on the specified activation function using appropriate initialization methods
	/// </summary>	
	void InitWeight( ActivationType activation )
	{
		var allSizeList = new List<int> { InputSize };
		allSizeList.AddRange( HiddenSizeList );
		allSizeList.Add( OutputSize );

		for( int i = 1; i < allSizeList.Count; i++ )
		{
			double scale = 0.01;
			if( activation == ActivationType.Relu  )
				scale = Math.Sqrt( 2.0 / allSizeList[ i - 1 ] ); // ReLU (He initialization)
			else if( activation == ActivationType.Sigmoid )
				scale = Math.Sqrt( 1.0 / allSizeList[ i - 1 ] ); // Sigmoid (Xavier initialization)
			//else if( double.TryParse( weightInitStd, out double customScale ) )
			//	scale = customScale;

			var normalDist = new Normal( 0.0, scale );
			Params[ "W" + i ] = Matrix<double>.Build.Random( allSizeList[ i - 1 ], allSizeList[ i ], normalDist );
			Params[ "b" + i ] = Vector<double>.Build.Dense( allSizeList[ i ], 0.0 );
		}
	}

	/// <summary>
	/// Perform a forward pass through the network to compute the output predictions for the given input data.
	/// </summary>
	/// <param name="x">Input data matrix</param>
	/// <returns>Output predictions matrix</returns>
	Matrix<double> Predict( Matrix<double> x )
	{
		this.Layers.ForEach( l => x = ForwardPass( l.Layer, x ) );

		return x;		
	}

	/// <summary>
	/// Compute the loss (cost) of the network's predictions compared to the true labels, including weight decay regularization if specified.
	/// </summary>
	/// <param name="x">Input data matrix</param>
	/// <param name="t">True labels matrix</param>
	/// <returns>Loss value</returns>
	public double Loss( Matrix<double> x, Matrix<double> t )
	{
		var y = Predict( x );

		double weightDecay = 0.0;

		for( int i = 1; i <= HiddenLayerNum + 1; i++ )
		{
			var w = ( Matrix<double> )Params[ "W" + i ];
			double sumSqr = w.Enumerate().Sum( val => val * val );

			weightDecay += 0.5 * WeightDecayLambda * sumSqr;
		}

		return LastLayer.Forward( y, t ) + weightDecay;
	}

	/// <summary>
	/// Calculate the accuracy of the network's predictions by comparing the predicted class labels to the true labels
	/// </summary>
	/// <param name="x">Input data matrix</param>
	/// <param name="t">True labels matrix</param>
	/// <returns>Accuracy value</returns>
	public double Accuracy( Matrix<double> x, Matrix<double> t )
	{
		var y = Predict( x );
		int batchSize = y.RowCount;
		int correct = 0;

		for( int i = 0; i < batchSize; i++ )
		{
			int yPred = y.Row( i ).MaximumIndex();
			int tIndex = t.ColumnCount == 1 ? ( int )t[ i, 0 ] : t.Row( i ).MaximumIndex();

			if( yPred == tIndex ) correct++;			
		}

		return ( double )correct / batchSize;
	}

	/// <summary>
	/// Update the network's parameters using the computed gradients and the optimizer.
	/// </summary>
	/// <param name="x">Input data matrix</param>
	/// <param name="t">True labels matrix</param>	
	public void Update( Matrix<double> x, Matrix<double> t )
	{
		var grads = Gradient( x, t ); // compute gradients via backprop
				
		Learn( grads ); // Convert mixed object dictionaries to strictly Matrix<double>
	}

	/// <summary>
	///  
	/// </summary>
	/// <param name="grads"></param>	
	void Learn( Dictionary<string, object> grads )
	{
		var matrixParams = new Dictionary<string, Matrix<double>>();
		var matrixGrads = new Dictionary<string, Matrix<double>>();

		// Convert all parameters to Matrix<double> format for the optimizer, handling both Vector and Matrix types
		foreach( var kvp in this.Params ) 
		{
			matrixParams[ kvp.Key ] = kvp.Value switch {
				Vector<double> v => v.ToRowMatrix(),
				Matrix<double> m => m,
				_ => throw new InvalidOperationException( "Unsupported parameter type." )
			};

			matrixGrads[ kvp.Key ] = grads[ kvp.Key ] switch {
				Vector<double> v => v.ToRowMatrix(),
				Matrix<double> m => m,
				_ => throw new InvalidOperationException( "Unsupported gradient type." )
			};
		}

		this.optimizer.Update( matrixParams, matrixGrads ); // Perform the update natively as matrices

		foreach( var kvp in matrixParams ) // Copy the updated parameters back into the original Params dictionary
		{
			if( this.Params[ kvp.Key ] is Vector<double> origVector )
			{
				kvp.Value.Row( 0 ).CopyTo( origVector ); // In-place copy the updated row matrix back into the original vector
			}
			else if( this.Params[ kvp.Key ] is Matrix<double> origMatrix )
			{
				if( !ReferenceEquals( origMatrix, kvp.Value ) )
				{
					kvp.Value.CopyTo( origMatrix ); // If the optimizer created a new matrix reference, copy it back into the original
				}
			}
		}
	}


	/// <summary>
	/// Run backpropagation to compute the gradients of the loss with respect to all weights and biases in the network,
	/// including weight decay regularization for the weight gradients.
	/// </summary>
	/// <param name="x">Input data matrix</param>
	/// <param name="t">True labels matrix</param>
	/// <returns>Dictionary containing gradients for all weights and biases</returns>
	Dictionary<string, object> Gradient( Matrix<double> x, Matrix<double> t )
	{		
		Loss( x, t ); // Forward pass
				
		Matrix<double> dout = LastLayer.Backward( 1.0 ); // Backward pass

		for( int i = Layers.Count - 1; i >= 0; i-- )
		{
			dout = BackwardPass( Layers[ i ].Layer, dout );
		}

		var grads = new Dictionary<string, object>();

		for( int i = 1; i <= HiddenLayerNum + 1; i++ )
		{
			var affineName = "Affine" + i;
			var affineLayer = ( Affine )Layers.First( l => l.Name == affineName ).Layer;

			var w = (Matrix<double>)this.Params[ "W" + i ];
						
			var dWWithDecay = affineLayer.dW + w.Multiply( WeightDecayLambda ); // W_grad = dW + lambda * W

			grads[ "W" + i ] = dWWithDecay;
			grads[ "b" + i ] = affineLayer.dB;
		}

		return grads;
	}

	// --- Duck Typing via Pattern Matching --- 

	static Matrix<double> ForwardPass( object layer, Matrix<double> x ) => layer switch 
	{
		Affine a => a.Forward( x ),
		Relu r => r.Forward( x ),
		Sigmoid s => s.Forward( x ),
		_ => throw new InvalidOperationException( $"Unknown layer type: {layer.GetType().Name}" )
	};

	static Matrix<double> BackwardPass( object layer, Matrix<double> dout ) => layer switch 
	{
		Affine a => a.Backward( dout ),
		Relu r => r.Backward( dout ),
		Sigmoid s => s.Backward( dout ),
		_ => throw new InvalidOperationException( $"Unknown layer type: {layer.GetType().Name}" )
	};

	#region obsolete
	//public MultiLayerNet( int inputSize, List<int> hiddenSizeList, int outputSize,
	//					  string activation = "relu", string weightInitStd = "relu", double weightDecayLambda = 0.0 )
	//{
	//	this.InputSize = inputSize;
	//	this.HiddenSizeList = hiddenSizeList;
	//	this.OutputSize = outputSize;
	//	this.HiddenLayerNum = hiddenSizeList.Count;
	//	this.WeightDecayLambda = weightDecayLambda;

	//	InitWeight( weightInitStd );

	//	for( int i = 1; i <= HiddenLayerNum; i++ )
	//	{
	//		var w = ( Matrix<double> )Params[ "W" + i ];
	//		var b = ( Vector<double> )Params[ "b" + i ];
	//		layersList.Add( ("Affine" + i, new Affine( w, b )) );

	//		if( activation.Equals( "relu", StringComparison.OrdinalIgnoreCase ) )
	//			layersList.Add( ("Activation_function" + i, new Relu()) );
	//		else if( activation.Equals( "sigmoid", StringComparison.OrdinalIgnoreCase ) )
	//			layersList.Add( ("Activation_function" + i, new Sigmoid()) );
	//	}

	//	int lastIdx = HiddenLayerNum + 1;
	//	var wLast = ( Matrix<double> )Params[ "W" + lastIdx ];
	//	var bLast = ( Vector<double> )Params[ "b" + lastIdx ];
	//	layersList.Add( ("Affine" + lastIdx, new Affine( wLast, bLast )) );
	//}
	//void InitWeight( string weightInitStd )
	//{
	//	var allSizeList = new List<int> { InputSize };
	//	allSizeList.AddRange( HiddenSizeList );
	//	allSizeList.Add( OutputSize );

	//	for( int i = 1; i < allSizeList.Count; i++ )
	//	{
	//		double scale = 0.01;
	//		if( weightInitStd.Equals( "relu", StringComparison.OrdinalIgnoreCase ) || weightInitStd.Equals( "he", StringComparison.OrdinalIgnoreCase ) )
	//			scale = Math.Sqrt( 2.0 / allSizeList[ i - 1 ] ); // ReLU (He initialization)
	//		else if( weightInitStd.Equals( "sigmoid", StringComparison.OrdinalIgnoreCase ) || weightInitStd.Equals( "xavier", StringComparison.OrdinalIgnoreCase ) )
	//			scale = Math.Sqrt( 1.0 / allSizeList[ i - 1 ] ); // Sigmoid (Xavier initialization)
	//		else if( double.TryParse( weightInitStd, out double customScale ) )
	//			scale = customScale;

	//		var normalDist = new Normal( 0.0, scale );
	//		Params[ "W" + i ] = Matrix<double>.Build.Random( allSizeList[ i - 1 ], allSizeList[ i ], normalDist );
	//		Params[ "b" + i ] = Vector<double>.Build.Dense( allSizeList[ i ], 0.0 );
	//	}
	//}
	#endregion
}