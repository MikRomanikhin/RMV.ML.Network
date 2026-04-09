using System.Text;

using RMV.ML.Network.Common;

namespace RMV.ML.Network.Domain;

/// <summary>
/// Initializes a new Node with the specified ID, learning rate, activation function, and optional momentum. 
/// </summary>   
public class Node( int id, AppSettings settings, IActivation? activation = null ) 
{

	#region State --------------------------------------------------------------		
	
	double bias, derivative, delta, sum = 0; 

	readonly IActivation activation = activation ?? new Relu();

	/// <summary>
	/// In/Out Synapses
	/// </summary>
	readonly List<Edge> InEdges = [], OutEdges = [];

	#endregion


	#region Properties ---------------------------------------------------------

	/// <summary>
	/// Node ID
	/// </summary>
	public int ID { get; } = id;

	/// <summary>
	/// The output value after activation function is applied
	/// </summary>
	public double Value { get; set; } = 0;

	/// <summary>
	/// The raw weighted sum before activation (used by Softmax output layer)
	/// </summary>
	public double RawValue { get; set; } = 0;	

	/// <summary>
	/// Learning Rate
	/// </summary>
	public double Rate { get; } = settings.Rate;

	/// <summary>
	/// Momentum parameter
	/// </summary>
	public double Momentum { get; } = settings.Momentum;

	/// <summary>
	/// The error to be minimized
	/// </summary>
	public double Error { get; private set; } = 0;	

   #endregion


   #region Connect ------------------------------------------------------------

   /// <summary>
   /// Connects this neuron to all neurons in the target Layer
   /// </summary>
   /// <param name="target">The Layer to which this neuron needs to be connected</param>
   internal void Connect( BaseLayer target )
   {
      foreach( var node in target.Nodes )
      {
         var edge = new Edge( this, node );

         this.OutEdges.Add( edge );

         node.InEdges.Add( edge );
      }
   }

	#endregion


	#region Initialise ---------------------------------------------------------

	/// <summary>
	/// Box-Mueller initialization (optimal for ReLU, prevents dying neurons)
	/// </summary>
	internal void BmInitialize()
	{
		if( this.InEdges.Count == 0 ) return;

		double std = Math.Sqrt( 2.0 / this.InEdges.Count );

		this.InEdges.ForEach( s => s.Weight = Tools.GetGaussian() * std );		

		this.bias = 0;
	}

	/// <summary>
	/// Nguen-Widrow initialisation 
	/// </summary>
	internal void NwInitialize()
	{
		if( this.InEdges.Count == 0 ) return;

		double rms = 0d;
		foreach( var s in this.InEdges ) // initialize synapse weights with random values
		{
			s.Weight = Tools.GetRandom();
			rms += s.Weight * s.Weight;
		}

		rms = Math.Sqrt( 1d / rms ); // calculate the root mean square of the weights

		double sum = 0d;
		foreach( var s in this.InEdges ) // scale the weights by the factor of 2 * rms and calculate the sum of weights
		{
			s.Weight *= 2d * rms;
			sum += s.Weight;
		}

		this.bias = Tools.GetRandom() - sum * 0.5; // initialize bias with a random value adjusted by the sum of weights
	}

	#endregion


	#region Propagate ----------------------------------------------------------

	/// <summary>
	/// Values are propagated to this neuron via source synapses.
	/// </summary>
	internal void Forward()
	{		
		this.RawValue = this.bias + this.InEdges.Sum( edge => edge.WeightedSourceValue );

		this.Value = this.activation.Function( this.RawValue );

		this.derivative = this.activation.Derivative( this.Value );
	}	

	#endregion


	#region Error BackPropagation ----------------------------------------------
		
	/// <summary>
	/// Output layer error (Softmax + Cross-Entropy)
	/// </summary>
	/// <param name="target">desired output (1 for correct class, 0 otherwise)</param>
	/// <returns>cross-entropy loss for this node</returns>
	internal double Backward( double target )
	{		
		this.Error = target - this.Value;  // gradient = ( t - o ) for softmax + cross-entropy

		this.sum += this.Error;  //  

		this.InEdges.ForEach( edge => edge.Learn() );		

		return -target * Math.Log( Math.Max( this.Value, 1e-15 ) );
	}
	

	/// <summary>
	/// Hidden Layer error
	/// </summary>
	internal void Backward()
   {
		this.Error = this.OutEdges.Sum( edge => edge.WeightedTargetError ) * this.derivative;
	
      this.sum += this.Error;

      this.InEdges.ForEach( edge => edge.Learn() );
   }

	#endregion


	#region Update -------------------------------------------------------------

	/// <summary>
	/// Weights and Bias update
	/// </summary>
	internal void Update( int batchSize )
	{
		this.InEdges.ForEach( edge => edge.Update( batchSize ) );

		double gradient = this.sum / batchSize;

		this.delta = gradient * this.Rate + this.delta * this.Momentum;
		this.bias += this.delta;

		this.sum = 0;
	}

	#endregion


	#region ToString -----------------------------------------------------------

	public override string ToString()
   {
      var sb = new StringBuilder();

      sb.Append( $"ID={this.ID} error={this.Error} value={this.Value} bias={this.bias} weights=[" );

		for( int i = 0; i < this.InEdges.Count; i++ )
		{
			if( i > 0 ) sb.Append( ", " );
			sb.Append( this.InEdges[ i ] );
		}

		sb.Append( ']' );

		return sb.ToString();
   }

	#endregion

}