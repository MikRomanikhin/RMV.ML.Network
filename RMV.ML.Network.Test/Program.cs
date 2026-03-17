//using NLog;
using System.Diagnostics;

using RMV.ML.Network.Common;
using RMV.ML.Network.Domain;

namespace RMV.ML.Network.Test;

/// <summary>
/// Neural Network Test Application
/// </summary>
class Program
{
	//static Logger logger = LogManager.GetCurrentClassLogger();	

	/// <summary>
	/// Main entry point
	/// </summary>	
	static void Main()
	{
		var timer = new Stopwatch();
		timer.Start();

		var settings = ConfigManager.GetRoot<AppSettings>() ?? throw new Exception( "Failed to load configuration" );					

		var parser = new Parser( settings.Output );
		var trainSet = parser.Run( File.ReadAllLines( settings.Train ) );
		var testSet = parser.Run( File.ReadAllLines( settings.Test ) );

		var network = new Net( settings ) { Learning = LearningType.Batch };
		network.OnReport += ( s, e ) => Console.WriteLine( e.Message + $" Time={timer.Elapsed:hh\\:mm\\:ss}" );
		network.Connect();
		network.Initialize();
		
		string message = $"Hidden:{settings.Hidden.Join()} Iterations:{settings.Iterations} Rate:{settings.Rate} Momentum:{settings.Momentum} Time:{timer.Elapsed.TotalSeconds:f2} sec";
		//logger.Info( message );
		Console.WriteLine( message );		

		network.Train( trainSet, testSet );
		          
		//File.WriteAllText( settings.Errors, errCsv.Join() );
		Console.ReadLine();
	}	
	
}
