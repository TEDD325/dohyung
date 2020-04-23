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
import org.edgexfoundry.domain.meta.ScheduleEvent;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class ScheduleEventDao {

  @Autowired
  private ScheduleEventRepository repos;

  @Autowired
  private DeviceReportRepository reportRepos;

  @Autowired
  private DeviceServiceRepository deviceServiceRepos;

  public ScheduleEvent getByIdOrName(ScheduleEvent scheduleEvent) {
    if (scheduleEvent == null)
      return null;
    if (scheduleEvent.getId() != null)
      return repos.findOne(scheduleEvent.getId());
    return repos.findByName(scheduleEvent.getName());
  }

  public boolean isScheduleEventAssociatedToDeviceReport(ScheduleEvent scheduleEvent) {
    if (scheduleEvent != null)
      return !reportRepos.findByEvent(scheduleEvent.getName()).isEmpty();
    return false;
  }

  public List<Asset> getAffectedService(ScheduleEvent event) {
    ArrayList<Asset> list = new ArrayList<>();
    if (event != null) {
      DeviceService ds = deviceServiceRepos.findByName(event.getService());
      if (ds != null)
        list.add(ds);
    }
    return list;
  }
}
