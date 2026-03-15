using System.Text;

using RMV.ML.Network.Common;

namespace RMV.ML.Network.Domain;

/// <summary>
/// Initializes a new Node with the specified ID, learning rate, activation function, and optional momentum. 
/// </summary>   
public class Node( int id, AppSettings settings, IActivation? activation = null )
{

	#region State --------------------------------------------------------------

	double sum = 0; // bias, error;
	double derivative, Delta; // for momentum

	readonly IActivation activation = activation ?? new Relu();

	/// <summary>
	/// In/Out Synapses
	/// </summary>
	readonly List<Edge> In = [], Out = [];

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
	/// The bias value added to the weighted sum of inputs before applying the activation function.
	/// </summary>
	double Bias { get; set; } = 0;   	

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
   public void Connect( Layer target )
   {
      foreach( var node in target.Nodes )
      {
         var edge = new Edge( this, node );

         this.Out.Add( edge );

         node.In.Add( edge );
      }
   }

	#endregion


	#region Initialise ---------------------------------------------------------

	/// <summary>
	/// He initialization (optimal for ReLU, prevents dying neurons)
	/// </summary>
	public void Initialize()
	{
		if( this.In.Count == 0 ) return;

		double std = Math.Sqrt( 2.0 / this.In.Count );

		this.In.ForEach( s => s.Weight = Tools.GetGaussian() * std );		

		this.Bias = 0;
	}

	/// <summary>
	/// Nguen-Widrow initialisation 
	/// </summary>
	public void NwInitialize()
	{
		if( this.In.Count == 0 ) return;

		double rms = 0d;
		foreach( var s in this.In ) // initialize synapse weights with random values
		{
			s.Weight = Tools.GetRandom();
			rms += s.Weight * s.Weight;
		}

		rms = Math.Sqrt( 1d / rms ); // calculate the root mean square of the weights

		double sum = 0d;
		foreach( var s in this.In ) // scale the weights by the factor of 2 * rms and calculate the sum of weights
		{
			s.Weight *= 2d * rms;
			sum += s.Weight;
		}

		this.Bias = Tools.GetRandom() - sum * 0.5; // initialize bias with a random value adjusted by the sum of weights
	}

	#endregion


	#region Propagate ----------------------------------------------------------

	/// <summary>
	/// Values are propagated to this neuron via source synapses.
	/// </summary>
	public void Forward()
	{
		//this.RawValue = this.In.Sum( e => e.WeightedSourceValue ) + this.Bias;
		double raw = this.Bias;
		for( int i = 0; i < this.In.Count; i++ ) raw += this.In[ i ].WeightedSourceValue;
		this.RawValue = raw;

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
	public double Back( double target )
	{		
		this.Error = target - this.Value;  // gradient = ( t - o ) for softmax + cross-entropy

		this.sum += this.Error;  //  

		this.In.ForEach( edge => edge.Learn() );		

		return -target * Math.Log( Math.Max( this.Value, 1e-15 ) );
	}
	

	/// <summary>
	/// Hidden Layer error
	/// </summary>
	public void Back()
   {
		//this.Error = this.Out.Sum( edge => edge.WeightedTargetError ) * this.derivative;
		double errorSum = 0;
      for( int i = 0; i < this.Out.Count; i++ ) errorSum += this.Out[ i ].WeightedTargetError;
      this.Error = errorSum * this.derivative;

      this.sum += this.Error;

      this.In.ForEach( edge => edge.Learn() );
   }

	#endregion


	#region Update -------------------------------------------------------------

	/// <summary>
	/// Weights and Bias update
	/// </summary>
	public void Update( int batchSize )
	{
		this.In.ForEach( edge => edge.Update( batchSize ) );

		double gradient = this.sum / batchSize;

		this.Delta = gradient * this.Rate + this.Delta * this.Momentum;
		this.Bias += this.Delta;

		this.sum = 0;
	}

	#endregion


	#region ToString -----------------------------------------------------------

	public override string ToString()
   {
      var sb = new StringBuilder();

      sb.Append( $"ID={this.ID} error={this.Error} value={this.Value} bias={this.Bias} weights=[" );

		for( int i = 0; i < this.In.Count; i++ )
		{
			if( i > 0 ) sb.Append( ", " );
			sb.Append( this.In[ i ] );
		}

		sb.Append( ']' );

		return sb.ToString();
   }

   #endregion

}