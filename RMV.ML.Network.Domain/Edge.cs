namespace RMV.ML.Network.Domain;

/// <summary>
/// Directed weighted connection between Source and Target neurons.
/// </summary>	
/// <param name="from">The source neuron feeding this Edge</param>
/// <param name="to">The target neuron fed by this Edge</param>
sealed class Edge( Node from, Node to )
{	
	double sum = 0; // Sum of weight updates for batch learning 
	double delta;   // Delta weight update 	

	/// <summary>
	/// The weight of the Synapse.
	/// </summary>
	internal double Weight { get; set; } = 0;

	/// <summary>
	/// Helpers
	/// </summary>
	internal double WeightedTargetError => this.Weight * to.Error;
	internal double WeightedSourceValue => this.Weight * from.Value;
			

	/// <summary>
	/// Accumulates the product of the source neuron's value and the target neuron's error for batch learning.
	/// </summary>
	internal void Learn() => this.sum += from.Value * to.Error;
   

   /// <summary>
   /// Weight adjustment
   /// </summary> 
	internal void Update( int batchSize )
	{
		double gradient = this.sum / batchSize;

		this.delta = gradient * from.Rate + this.delta * from.Momentum;
		this.Weight += this.delta;

		this.sum = 0;
	}   
		   

   public override string ToString() => $"from:{from.ID} to:{to.ID} weight:{this.Weight}";     

}
