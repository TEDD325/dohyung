/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.dao;

import java.util.ArrayList;
import java.util.List;

import org.edgexfoundry.domain.meta.Asset;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.domain.meta.Schedule;
import org.edgexfoundry.domain.meta.ScheduleEvent;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class ScheduleDao {

  @Autowired
  private ScheduleRepository repos;

  @Autowired
  private ScheduleEventRepository scheduleEventRepos;

  @Autowired
  private DeviceServiceRepository deviceServiceRepository;

  public Schedule getByIdOrName(Schedule schedule) {
    if (schedule == null)
      return null;
    if (schedule.getId() != null)
      return repos.findOne(schedule.getId());
    return repos.findByName(schedule.getName());
  }

  public boolean isScheduleAssociatedToScheduleEvent(Schedule schedule) {
    if (schedule != null)
      return !getAssociatedScheduleEvents(schedule).isEmpty();
    return false;
  }

  public List<Asset> getAffectedServices(Schedule schedule) {
    // for each schedule event get the associated service
    ArrayList<Asset> assets = new ArrayList<>();
    List<ScheduleEvent> list = getAssociatedScheduleEvents(schedule);
    for (ScheduleEvent e : list) {
      String name = e.getService();
      if (name != null) {
        DeviceService service = deviceServiceRepository.findByName(name);
        if (service != null)
          assets.add(service);
      }
    }
    return assets;
  }

  private List<ScheduleEvent> getAssociatedScheduleEvents(Schedule schedule) {
    if (schedule != null)
      return scheduleEventRepos.findBySchedule(schedule.getName());
    return new ArrayList<>();
  }
}
