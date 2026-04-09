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
	
	/// <summary>
	/// Weight matrices indexed by layer (1-based: index 0 is unused).
	/// </summary>
	Matrix<double>[] Weights { get; }

	/// <summary>
	/// Bias vectors indexed by layer (1-based: index 0 is unused).
	/// </summary>
	Vector<double>[] Biases { get; }

	/// <summary>
	/// Ordered list of all layers (affine + activation) for forward/backward traversal.
	/// </summary>
	readonly List<ILayer> Layers = [];

	/// <summary>
	/// Affine layers indexed by layer number (1-based: index 0 is unused) for gradient extraction.
	/// </summary>
	readonly Affine[] AffineLayers;	

	SoftmaxWithLoss LastLayer { get; } = new();

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

		int totalLayers = HiddenLayerNum + 1;
		this.Weights = new Matrix<double>[ totalLayers + 1 ]; // 1-based indexing
		this.Biases = new Vector<double>[ totalLayers + 1 ];
		this.AffineLayers = new Affine[ totalLayers + 1 ];

		this.optimizer = BuildOptimizer( settings );

		InitWeight( settings.Activation );

		for( int i = 1; i <= HiddenLayerNum; i++ )
		{
			var affine = new Affine( this.Weights[ i ], this.Biases[ i ] );
			this.AffineLayers[ i ] = affine;
			this.Layers.Add( affine );

			if( settings.Activation == ActivationType.Relu )
				this.Layers.Add( new Relu() );
			else if( settings.Activation == ActivationType.Sigmoid )
				this.Layers.Add( new Sigmoid() );
		}

		int lastIdx = this.HiddenLayerNum + 1;
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
						
			this.Weights[ i ] = Matrix<double>.Build.Random( allSizeList[ i - 1 ], allSizeList[ i ], new Normal( 0.0, scale ) );
			this.Biases[ i ] = Vector<double>.Build.Dense( allSizeList[ i ], 0.0 );			
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

		for( int i = 1; i <= HiddenLayerNum + 1; i++ )
		{			
			double sumSqr = this.Weights[ i ].Enumerate().Sum( val => val * val );

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
		var (dW, dB) = Gradient( x, t ); // compute gradients via backprop

		this.optimizer.Update( this.Weights, this.Biases, dW, dB ); // update parameters directly
	}

	/// <summary>
	/// Run backpropagation to compute the gradients of the loss with respect to all weights and biases in the network,
	/// including weight decay regularization for the weight gradients.
	/// </summary>
	/// <param name="x">Input data matrix</param>
	/// <param name="t">True labels matrix</param>
	/// <returns>Tuple of weight gradients and bias gradients arrays (1-based indexing)</returns>
	(Matrix<double>[], Vector<double>[]) Gradient( Matrix<double> x, Matrix<double> t )
	{
		Loss( x, t ); // Forward pass

		Matrix<double> dout = LastLayer.Backward( 1.0 ); // Backward pass

		for( int i = Layers.Count - 1; i >= 0; i-- )
		{
			dout = Layers[ i ].Backward( dout );
		}

		int totalLayers = HiddenLayerNum + 1;
		var dW = new Matrix<double>[ totalLayers + 1 ];
		var dB = new Vector<double>[ totalLayers + 1 ];

		for( int i = 1; i <= totalLayers; i++ )
		{
			dW[ i ] = AffineLayers[ i ].dW + this.Weights[ i ].Multiply( WeightDecayLambda ); // W_grad = dW + lambda * W
			dB[ i ] = AffineLayers[ i ].dB;
		}

		return (dW, dB);
	}	
	
}