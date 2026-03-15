namespace RMV.ML.Network.Domain;

/// <summary>
/// Directed weighted connection between Source and Target neurons.
/// </summary>	
/// <param name="from">The source neuron feeding this Synapse</param>
/// <param name="to">The target neuron fed by this Synapse</param>
sealed class Edge( Node from, Node to )
{	
	double sum = 0;
	double delta; // Delta weight update 	

	/// <summary>
	/// The weight of the Synapse.
	/// </summary>
	public double Weight { get; set; } = 0;

	/// <summary>
	/// Helpers
	/// </summary>
	public double WeightedTargetError => this.Weight * to.Error;
	public double WeightedSourceValue => this.Weight * from.Value;
			

	/// <summary>
	/// Calculates Delta weight update
	/// </summary>
	public void Learn()
   {
      double delta = to.Error * from.Value;

      this.sum += delta;
   }

   /// <summary>
   /// Weight adjustment
   /// </summary> 
	public void Update( int batchSize )
	{
		double gradient = this.sum / batchSize;

		this.delta = gradient * from.Rate + this.delta * from.Momentum;
		this.Weight += this.delta;

		this.sum = 0;
	}   
		   

   public override string ToString() => $"from:{from.ID} to:{to.ID} weight:{this.Weight}";     

}
