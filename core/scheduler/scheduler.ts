import { HierarchicalCache } from '../cache/hierarchical_cache.js';
import { AiloStorageFabric } from '../storage/storage_fabric.js';

export interface ScheduledTask {
  id: string;
  tensorName: string;
  layerIndex: number;
  priority: number;
  sourceDriveId: string;
  sizeBytes: number;
  status: 'PENDING' | 'TRANSFERRING' | 'READY' | 'COMPLETED';
}

export class AiloScheduler {
  private fabric: AiloStorageFabric;
  private cache: HierarchicalCache;
  private queue: ScheduledTask[] = [];

  constructor(fabric: AiloStorageFabric, cache: HierarchicalCache) {
    this.fabric = fabric;
    this.cache = cache;
  }

  public scheduleTensorTransfer(
    tensorName: string,
    layerIndex: number,
    priority: number,
    sourceDriveId: string,
    sizeBytes: number
  ): ScheduledTask {
    const task: ScheduledTask = {
      id: `task_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      tensorName,
      layerIndex,
      priority,
      sourceDriveId,
      sizeBytes,
      status: 'PENDING',
    };

    this.queue.push(task);
    // Sort tasks by priority (highest first)
    this.queue.sort((a, b) => b.priority - a.priority);
    return task;
  }

  public getPendingCount(): number {
    return this.queue.filter(t => t.status === 'PENDING').length;
  }
}
