using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

using RMV.ML.Network.Common;

namespace RMV.ML.MNIST.CS;

/// <summary>
/// Fully connected deep neural network with arbitrary hidden layers.
/// </summary>
public class MultiLayerNet
{
	readonly int inputSize, outputSize; // input and output dimensions
	readonly List<int> HiddenSizeList;  // list of hidden layer sizes
	readonly int hiddenLayerNum;        // number of hidden layers
	readonly double weightDecayLambda;  // weight decay coefficient

	/// <summary>
	/// Weight matrices indexed by layer 
	/// </summary>
	Matrix<double>[] Weights { get; }

	/// <summary>
	/// Bias vectors indexed by layer 
	/// </summary>
	Vector<double>[] Biases { get; }

	/// <summary>
	/// Ordered list of all layers (affine + activation) for forward/backward traversal.
	/// </summary>
	readonly List<ILayer> Layers = [];

	/// <summary>
	/// Affine layers indexed by layer number for gradient extraction.
	/// </summary>
	readonly Affine[] AffineLayers;

	SoftmaxWithLoss LastLayer { get; } = new();

	readonly IOptimizer optimizer;

	/// <summary>
	/// Construct a multi-layer fully connected neural network based on the provided settings.
	/// </summary>
	public MultiLayerNet( AppSettings settings )
	{
		this.inputSize = settings.Input;
		this.HiddenSizeList = [ .. settings.Hidden ];
		this.outputSize = settings.Output;
		this.hiddenLayerNum = settings.Hidden.Length;
		this.weightDecayLambda = settings.Decay;

		int totalLayers = hiddenLayerNum + 1;
		this.Weights = new Matrix<double>[ totalLayers ];
		this.Biases = new Vector<double>[ totalLayers ];
		this.AffineLayers = new Affine[ totalLayers ];

		this.optimizer = BuildOptimizer( settings );

		InitWeight( settings.Activation );

		for( int i = 0; i < hiddenLayerNum; i++ )
		{
			var affine = new Affine( this.Weights[ i ], this.Biases[ i ] );
			this.AffineLayers[ i ] = affine;
			this.Layers.Add( affine );

			if( settings.Activation == ActivationType.Relu )
				this.Layers.Add( new Relu() );
			else if( settings.Activation == ActivationType.Sigmoid )
				this.Layers.Add( new Sigmoid() );
		}

		int lastIdx = hiddenLayerNum;
		var lastAffine = new Affine( this.Weights[ lastIdx ], this.Biases[ lastIdx ] );
		this.AffineLayers[ lastIdx ] = lastAffine;
		this.Layers.Add( lastAffine );
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
		var allSizeList = new List<int> { inputSize };
		allSizeList.AddRange( HiddenSizeList );
		allSizeList.Add( outputSize );

		for( int i = 0; i < allSizeList.Count - 1; i++ )
		{
			double scale = 0.01;
			if( activation == ActivationType.Relu )
				scale = Math.Sqrt( 2.0 / allSizeList[ i ] ); // ReLU (He initialization)
			else if( activation == ActivationType.Sigmoid )
				scale = Math.Sqrt( 1.0 / allSizeList[ i ] ); // Sigmoid (Xavier initialization)
			//else if( double.TryParse( weightInitStd, out double customScale ) )
			//	scale = customScale;

			this.Weights[ i ] = Matrix<double>.Build.Random( allSizeList[ i ], allSizeList[ i + 1 ], new Normal( 0.0, scale ) );
			this.Biases[ i ] = Vector<double>.Build.Dense( allSizeList[ i + 1 ], 0.0 );
		}
	}

	/// <summary>
	/// Perform a forward pass through the network to compute the output predictions for the given input data.
	/// </summary>
	/// <param name="x">Input data matrix</param>
	/// <returns>Output predictions matrix</returns>
	Matrix<double> Predict( Matrix<double> x )
	{
		this.Layers.ForEach( l => x = l.Forward( x ) );

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

		for( int i = 0; i < Weights.Length; i++ )
		{
			double sumSqr = this.Weights[ i ].Enumerate().Sum( val => val * val );

			weightDecay += 0.5 * weightDecayLambda * sumSqr;
		}

		return LastLayer.Forward( y, t ) + weightDecay;
	}

	/// <summary>
	/// Calculate the accuracy of the network's predictions by comparing the predicted class labels to the true labels
	/// </summary>
	/// <param name="x">Input data matrix</param>
	/// <param name="t">True labels matrix</param>
	/// <returns>Accuracy value</returns>
	public (double, List<int>, List<int>) Accuracy( Matrix<double> x, Matrix<double> t )
	{
		var y = Predict( x );
		int batchSize = y.RowCount;
		int correct = 0;
		List<int> errors = [];
		List<int> indexes = [];

		for( int i = 0; i < batchSize; i++ )
		{
			int yPred = y.Row( i ).MaximumIndex();
			int tIndex = t.ColumnCount == 1 ? ( int )t[ i, 0 ] : t.Row( i ).MaximumIndex();

			if( yPred == tIndex ) correct++;  // count correct predictions
			else // store misclassified sample information 
			{
				errors.Add( yPred );  // store predicted label 
				indexes.Add( i );     // store image index 
			}		
		}

		return (( double )correct / batchSize, errors, indexes);
	}

	/// <summary>
	/// Update the network's parameters using the computed gradients and the optimizer.
	/// </summary>
	/// <param name="x">Input data matrix</param>
	/// <param name="t">True labels matrix</param>
	public void Update( Matrix<double> x, Matrix<double> t )
	{
		var (dW, dB) = Gradient( x, t ); // compute gradients via backprop

		this.optimizer.Update( this.Weights, this.Biases, dW, dB ); // update parameters directly
	}

	/// <summary>
	/// Run backpropagation to compute the gradients of the loss with respect to all weights and biases in the network,
	/// including weight decay regularization for the weight gradients.
	/// </summary>
	/// <param name="x">Input data matrix</param>
	/// <param name="t">True labels matrix</param>
	/// <returns>Tuple of weight gradients and bias gradients arrays (0-based indexing)</returns>
	(Matrix<double>[], Vector<double>[]) Gradient( Matrix<double> x, Matrix<double> t )
	{
		Loss( x, t ); // Forward pass

		Matrix<double> dout = LastLayer.Backward( 1.0 ); // Backward pass

		for( int i = Layers.Count - 1; i >= 0; i-- )
		{
			dout = Layers[ i ].Backward( dout );
		}

		var dW = new Matrix<double>[ AffineLayers.Length ];
		var dB = new Vector<double>[ AffineLayers.Length ];

		for( int i = 0; i < AffineLayers.Length; i++ )
		{
			dW[ i ] = AffineLayers[ i ].dW + this.Weights[ i ].Multiply( weightDecayLambda ); // W_grad = dW + lambda * W
			dB[ i ] = AffineLayers[ i ].dB;
		}

		return (dW, dB);
	}

}