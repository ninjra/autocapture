using System.Collections.ObjectModel;
using Autocapture.Shell.Models;

namespace Autocapture.Shell.ViewModels.DesignTime;

public sealed class DesignTimelineViewModel : ViewModelBase
{
    public ObservableCollection<TimelineItem> Items { get; } = new()
    {
        new TimelineItem
        {
            TimeLabel = "08:24",
            Title = "Daily planning",
            Summary = "Reviewed goals and blocked focus time.",
            Context = "Calendar"
        },
        new TimelineItem
        {
            TimeLabel = "09:10",
            Title = "API explorations",
            Summary = "Mapped response shapes for the timeline card view.",
            Context = "Docs"
        },
        new TimelineItem
        {
            TimeLabel = "11:42",
            Title = "Customer notes",
            Summary = "Collected feedback on what a clean shell should show.",
            Context = "Research"
        }
    };
}
