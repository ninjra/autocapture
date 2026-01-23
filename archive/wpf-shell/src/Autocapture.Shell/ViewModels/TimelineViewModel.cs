using System.Collections.ObjectModel;
using Autocapture.Shell.Models;

namespace Autocapture.Shell.ViewModels;

public sealed class TimelineViewModel : ViewModelBase
{
    public ObservableCollection<TimelineItem> Items { get; } = new();
}
