using System.Diagnostics;

using RMV.ML.Network.Common;
using RMV.ML.Network.Domain;

namespace RMV.ML.Network.Test;

/// <summary>
/// Neural Network Test Application
/// </summary>
class Program
{	
	/// <summary>
	/// Main entry point
	/// </summary>	
	static async Task Main()
	{
		var timer = new Stopwatch();
		timer.Start();

		var settings = ConfigManager.GetRoot<AppSettings>() ?? throw new Exception( "Failed to load configuration" );					

		var parser = new Parser( settings.Output );
		var trainSet = parser.Run( File.ReadAllLines( settings.TrainPath ) );
		var testSet = parser.Run( File.ReadAllLines( settings.TestPath ) );

		Console.WriteLine( $"Loaded data sets. Time:{timer.Elapsed.TotalSeconds:f2} sec" );

		var network = new Net( settings, timer );
		network.OnReport += ( s, e ) => Console.WriteLine( e.Message );
		network.Connect();
		network.Initialize();
		
		string message = $"Hidden:{settings.Hidden.Join()} Iterations:{settings.Iterations} Rate:{settings.Rate} Momentum:{settings.Momentum} Time:{timer.Elapsed.TotalSeconds:f2} sec";
		Console.WriteLine( message );

		switch( settings.Learning )
		{
			case LearningType.Online:
				await network.TrainOnline( trainSet, testSet );
				break;
			case LearningType.Batch:
				await network.TrainBatch( trainSet, testSet	);
				break;
			case LearningType.MiniBatch:
				await network.TrainMiniBatch( trainSet, testSet );
				break;
		}

		//File.WriteAllText( settings.Errors, errCsv.Join() );
		timer.Stop();
		Console.WriteLine( "Training completed. Time={timer.Elapsed:hh\\:mm\\:ss}" );
		Console.ReadLine();
	}	
	
}
